from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import Code, SinglefileData
from aiida.common.folders import Folder

from aiida_aenet.data.algorithm import AenetAlgorithm
from aiida_aenet.data.potentials import AenetPotential


class AenetTrainCalculation(CalcJob):
    """CalcJob implementation to run train.x with aiida-aenet.
    
    Parameters
    ----------
    code : Code
        A Code node that is linked to a build of train.x.
    algorithm : AenetAlgorithm
        A custom Node that contains the neural-network algorithm parameters,
        including element data, descriptor data, training method data,
        and the learning parameters.
    train_file : SinglefileData
        A SinglefileData node that contains the binary training set descriptor
        data produced from generate.x.
    
    Returns
    -------
    potential : AenetPotential
        A custom Node with the ANN potential file data, suitable for aiida-lammps
        and an aenet-lammps build of LAMMPS.
    """
    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """CalcJob I/O specification."""

        # TODO store training calc files in a dedicated directory

        super().define(spec)
        spec.input('code', valid_type=Code)
        spec.input('algorithm', valid_type=AenetAlgorithm)
        spec.input('train_file', valid_type=SinglefileData)
        spec.input(
            'metadata.options.resources',
            valid_type=dict,
            default={
                'num_machines': 1,
                'num_mpiprocs_per_machine': 1
            },
        )
        spec.input(
            'metadata.options.parser_name',
            valid_type=str,
            default='aenet.train',
        )
        spec.output('potential', valid_type=AenetPotential)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Setup the working directory and calculation procedure."""

        ann_potential_files = [
            f"{X}.nn" for X in self.inputs.algorithm.elements
        ]

        with folder.open("train.in", "w", encoding="utf8") as handle:
            handle.write(self.write_input())

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = "train.out"
        codeinfo.cmdline_params = ["train.in"]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [(
            self.inputs.train_file.uuid,
            self.inputs.train_file.filename,
            "train.dat",
        )]
        calcinfo.retrieve_list = ["train.out"] + ann_potential_files

        return calcinfo

    def write_input(self):
        """Write the train.x input file to the sandbox directory."""

        elements = self.inputs.algorithm.elements
        test_percent = self.inputs.algorithm.test_percent
        iterations = self.inputs.algorithm.epochs
        max_energy = self.inputs.algorithm.max_energy
        save_energies = self.inputs.algorithm.save_energies
        timing = self.inputs.algorithm.timing
        debug = self.inputs.algorithm.debug
        method = self.inputs.algorithm.train_method

        input_lines = [
            f"TRAININGSET train.dat",
            f"TESTPERCENT {test_percent}",
            f"ITERATIONS {iterations}",
            "",
            f"MAXENERGY {max_energy}",
            "",
        ]

        if save_energies: input_lines += ["SAVE_ENERGIES", ""]
        if timing: input_lines += ["TIMING"]
        if debug: input_lines += ["DEBUG"]

        # TODO specific to bfgs - generalize for different training methods

        input_lines += ["", "METHOD", method, ""]

        input_lines += [
            "NETWORKS",
            "! atom   network         hidden",
            "! types  file-name       layers  nodes:activation",
        ]

        for element, traits in elements.items():

            file = f"{element}.nn"

            network = traits['network']
            layers = len(network)

            network_str = " ".join([
                f"{layer['nodes']}:{layer['activation']}" for layer in network
            ])
            network_line = (
                f"{element:8s} {file:<15s} {layers:<7d} {network_str}")

            input_lines += [network_line]

        return "\n".join(input_lines)