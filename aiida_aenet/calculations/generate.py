from itertools import combinations_with_replacement

from aiida.orm import Code, List, SinglefileData, load_node
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.common.folders import Folder

from aiida_aenet.data.algorithm import AenetAlgorithm


class AenetGenerateCalculation(CalcJob):
    """CalcJob implementation to run generate.x with aiida-aenet.

    TODO xsf flag - check if files already provided
    
    Parameters
    ----------
    code : Code
        A Code node that is linked to a build of generate.x.
    reference : List
        A List node of StructureData PKs that make up the reference set.
    algorithm : AenetAlgorithm
        A custom Node that contains the neural-network algorithm parameters,
        including element data, descriptor data, training method data,
        and the learning parameters.

    Returns
    -------
    train_file : SinglefileData
        A SinglefileData node that contains the binary training set descriptor
        data produced from generate.x.
    """
    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """CalcJob I/O specification."""
        super().define(spec)
        spec.input('code', valid_type=Code)
        spec.input('reference', valid_type=List)
        spec.input('algorithm', valid_type=AenetAlgorithm)
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
            default='aenet.generate',
        )
        spec.output('train_file', valid_type=SinglefileData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Setup the working directory and calculation procedure."""

        self.write_xsfs(folder)
        self.write_setups(folder)

        with folder.open("generate.in", "w", encoding="utf8") as handle:
            handle.write(self.write_input())

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = "generate.out"
        codeinfo.cmdline_params = ["generate.in"]

        # TODO add generate.out & generate.time to retrieve list

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = ["train.dat"]

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

    def write_setups(self, folder):
        """Write aenet setup files to sandbox directory for each element."""

        descriptor = self.inputs.algorithm.descriptor
        elements = self.inputs.algorithm.elements
        r_min = self.inputs.algorithm.r_min

        for element in elements:

            stp_lines = [
                "DESCR", f"Setup for {element}", "END DESCR", "",
                f"ATOM {element}", "", f"ENV {len(elements)}"
            ]

            stp_lines += [*elements]

            stp_lines += [
                "",
                f"RMIN {r_min}d0",
                "",
            ]

            stp_lines += list(getattr(self, f"{descriptor['type']}")())

            with folder.open(f"{element}.stp", "w", encoding="utf8") as stp:
                stp.write("\n".join(stp_lines))

    def write_input(self):
        """Write the generate.x input file to the sandbox directory."""

        reference_list = self.inputs.reference.get_list()
        elements = self.inputs.algorithm.elements
        timing = self.inputs.algorithm.timing
        debug = self.inputs.algorithm.debug

        energies = []
        setups = []

        for element, traits in elements.items():

            energies.append(f"{element:3s} {traits['energy']}  ! eV")
            setups.append(f"{element:3s} {element}.stp")

        input_lines = [f"OUTPUT train.dat", ""]

        if debug: input_lines += ["DEBUG"]
        if timing: input_lines += ["TIMING"]

        input_lines += ["TYPES", f"{len(elements)}"]
        input_lines += energies
        input_lines += ["", "SETUPS"]
        input_lines += setups
        input_lines += ["", "FILES", f"{len(reference_list)}"]
        input_lines += [xsf for xsf in self.xsf_file_list]

        return "\n".join(input_lines)

    def behler(self):
        """Return the fingerprint strings for the Behler-Parrinnello symmetry functions."""

        symbols = [e.symbol for e in self.inputs.algorithm.elements]
        parameters = self.parameters

        Rc = parameters["cutoff"],
        g2_etas = parameters["G2"]["etas"]
        g4_etas = parameters["G4"]["etas"]
        g4_lambdas = parameters["G4"]["lambdas"]
        g4_zetas = parameters["G4"]["zetas"]

        # FIXME NEED TO GENERALIZE METHOD FOR MULTINARY ALLOYS

        radial_fingerprints = []

        for symbol in symbols:

            for eta in g2_etas:

                radial_fingerprints.append([
                    ('G', 2),
                    ('type2', symbol),
                    ('eta', eta),
                    ('Rs', 0.0000),
                    ('Rc', Rc),
                ])

        permutations = set(combinations_with_replacement(
            symbols, len(symbols)))

        angular_fingerprints = []

        for zeta in g4_zetas:

            for _lambda in g4_lambdas:

                for eta in g4_etas:

                    for (t2, t3) in permutations:

                        fingerprint = [
                            ('G', 4),
                            ('type2', t2),
                            ('type3', t3),
                            ('eta', eta),
                            ('lambda', _lambda),
                            ('zeta', zeta),
                            ('Rc', Rc),
                        ]

                angular_fingerprints += [fingerprint]

        fingerprints = radial_fingerprints + angular_fingerprints

        param_lines = '\n'.join([
            '  '.join(['{}={}'.format(k, v) for k, v in fingerprint])
            for fingerprint in fingerprints
        ])

        fingerprint_lines = [
            "SYMMFUNC type=Behler2011", f"{len(fingerprints)}",
            f"{param_lines}\n"
        ]

        return fingerprint_lines

    def chebyshev(self):
        """Return the fingerprint strings for the Chebyshev descriptors."""

        parameters = self.inputs.algorithm.descriptor["parameters"]

        parameters.setdefault("radial cutoff", 4.0)
        parameters.setdefault("radial n", 6)
        parameters.setdefault("angular cutoff", 4.0)
        parameters.setdefault("angular n", 2)

        fingerprint_lines = [
            "BASIS type=Chebyshev",
            f"radial_Rc = {parameters['radial cutoff']}",
            f"radial_N = {parameters['radial n']}",
            f"angular_Rc = {parameters['angular cutoff']}",
            f"angular_N = {parameters['angular n']}"
        ]

        return [
            f"{fingerprint_lines[0]}",
            "{0} {1} {2} {3}".format(*fingerprint_lines[1:])
        ]