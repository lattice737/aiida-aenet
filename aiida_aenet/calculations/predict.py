from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import Code, List, Dict, load_node
from aiida.common.folders import Folder

from aiida_aenet.data.algorithm import AenetAlgorithm
from aiida_aenet.data.potentials import AenetPotential


class AenetPredictCalculation(CalcJob):
    """CalcJob implementation to run predict.x with aiida-aenet

    TODO xsf flag - check if files already provided
    
    Parameters
    ----------
    code : Code
        A Code node that is linked to a build of predict.x.
    reference : List
        A List node of StructureData PKs that make up the reference set.
    algorithm : AenetAlgorithm
        A custom Node that contains the neural-network algorithm parameters,
        including element data, descriptor data, training method data,
        and the learning parameters.
    potential : AenetPotential
        A custom Node with the ANN potential file data, suitable for aiida-lammps
        and an aenet-lammps build of LAMMPS.
    """
    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """CalcJob I/O specification."""
        super().define(spec)
        spec.input('code', valid_type=Code)
        spec.input('reference', valid_type=List)
        spec.input('algorithm', valid_type=AenetAlgorithm)
        spec.input('potential', valid_type=AenetPotential)
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
            default="aenet.predict",
        )
        spec.output('results', valid_type=Dict)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Setup the working directory and calculation procedure."""

        # TODO store xsfs in dedicated directory

        self.write_xsfs(folder)
        self.write_potentials(folder)

        with folder.open("predict.in", "w", encoding="utf8") as handle:
            handle.write(self.write_input())

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdin_name = "predict.in"
        codeinfo.stdout_name = "predict.out"
        codeinfo.cmdline_params = ["predict.in"]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = ["predict.out"]

        return calcinfo

    def write_xsfs(self, folder):
        """Write calculation-agnostic xsf files to sandbox directory."""

        reference_list = self.inputs.reference.get_list()

        self.xsf_file_list = []

        for i, pk in enumerate(reference_list):

            pw = load_node(pk)
            structure = pw.inputs.structure
            energy = pw.res.energy
            forces = pw.outputs.output_trajectory.get_array('forces')[0]

            xsf = [
                f"# {structure.label}", "", f"# total energy = {energy} eV",
                "", "CRYSTAL", "PRIMVEC"
            ]

            xsf += ["{} {} {}".format(*v) for v in structure.cell]
            xsf += ["PRIMCOORD", f"{len(structure.sites)} 1"]

            for j, atom in enumerate(structure.sites):

                x, y, z = atom.position
                fx, fy, fz = forces[j]

                xsf += [
                    f"{atom.kind_name:<3s} {x: 18.12f} {y: 18.12f} {z: 18.12f}"
                    f" {fx: 18.12f} {fy: 18.12f} {fz: 18.12f}"
                ]

            padding = len(str(len(reference_list)))
            xsf_file = f"{str(i+1).zfill(padding)}.xsf"

            self.xsf_file_list.append(xsf_file)

            with folder.open(xsf_file, "w", encoding="utf8") as f:
                f.write("\n".join(xsf))

    def write_potentials(self, folder):
        """Write the binary potential data to sandbox directory files."""

        potential = self.inputs.potential

        for file in potential.get_external_files():

            binary = potential.get_object_content(file, mode='rb')

            with folder.open(file, 'wb') as handle:
                handle.write(binary)

    def write_input(self):
        """Write the predict.x input file to the sandbox directory."""

        reference_list = self.inputs.reference.get_list()
        elements = self.inputs.algorithm.elements
        predict_forces = self.inputs.algorithm.predict_forces
        predict_relax = self.inputs.algorithm.predict_relax
        timing = self.inputs.algorithm.timing

        input_lines = ["TYPES", f"{len(elements)}"]
        input_lines += elements.keys()
        input_lines += ["", "NETWORKS"]
        input_lines += [f"{X:3s}  {X}.nn" for X in elements]
        input_lines += [""]

        if predict_forces: input_lines += ["FORCES"]
        if predict_relax: input_lines += ["RELAX", predict_relax]
        if timing: input_lines += ["TIMING"]

        input_lines += ["", "FILES", f"{len(reference_list)}"]
        input_lines += [xsf for xsf in self.xsf_file_list]

        return "\n".join(input_lines)
