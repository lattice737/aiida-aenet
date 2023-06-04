import numpy as np

from aiida.orm import Dict
from aiida.common import CalcInfo, CodeInfo
from aiida.common.exceptions import ValidationError

from aiida_aenet.data.potentials import AenetPotential

from aiida_lammps.calculations.lammps.md_multi import MdMultiCalculation
from aiida_lammps.common.generate_structure import generate_lammps_structure


class AenetLammpsMdCalculation(MdMultiCalculation):
    """CalcJob implementation to run a multi-stage MD simulation with aiida-lammps.

    Parameters
    ----------
    md : MdMultiCalculation
        The input dictionary of MdMultiCalculation (aiida-lammps).
    potential : AenetPotential
        A custom Node with the ANN potential file data suitable for aiida-lammps
        and an aenet-lammps build of LAMMPS.
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(MdMultiCalculation, exclude=("potential", ))
        spec.input(
            "potential",
            valid_type=AenetPotential,
            help="aenet-lammps potential",
        )
        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default="aenet.simulate",
        )

    def prepare_for_submission(self, tempfolder):
        """Setup the working directory and calculation procedure."""

        # assert that the potential and structure have the same kind elements

        if self.inputs.potential.allowed_element_names is not None and not set(
                k.symbol for k in self.inputs.structure.kinds).issubset(
                    self.inputs.potential.allowed_element_names):
            raise ValidationError(
                "the structure and potential are not compatible (different kind elements)"
            )

        # Setup structure

        structure_txt, struct_transform = generate_lammps_structure(
            self.inputs.structure, self.inputs.potential.atom_style)

        with open(
                tempfolder.get_abs_path(self.options.cell_transform_filename),
                "w+b") as handle:
            np.save(handle, struct_transform)

        if "parameters" in self.inputs:
            parameters = self.inputs.parameters
        else:
            parameters = Dict()

        # Setup input parameters

        input_txt = self.create_main_input_content(
            parameter_data=parameters,
            potential_data=self.inputs.potential,
            kind_symbols=[kind.symbol for kind in self.inputs.structure.kinds],
            structure_filename=self._INPUT_STRUCTURE,
            trajectory_filename=self.options.trajectory_suffix,
            system_filename=self.options.system_suffix,
            restart_filename=self.options.restart_filename,
        )

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)

        with open(input_filename, "w") as infile:
            infile.write(input_txt)

        self.validate_parameters(parameters, self.inputs.potential)
        retrieve_list, retrieve_temporary_list = self.get_retrieve_lists()
        retrieve_list.extend([
            self.options.output_filename,
            self.options.cell_transform_filename,
        ])

        # prepare extra files if needed
        self.prepare_extra_files(tempfolder, self.inputs.potential)

        # =========================== dump to file =============================

        structure_filename = tempfolder.get_abs_path(self._INPUT_STRUCTURE)

        with open(structure_filename, "w") as infile:
            infile.write(structure_txt)

        potential = self.inputs.potential

        for file in potential.get_external_files():

            content = potential.get_object_content(file, mode='rb')
            file_path = tempfolder.get_abs_path(file)

            with open(file_path, "wb") as potential_file:
                potential_file.write(content)

        # ============================ calcinfo ================================

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = list(self._cmdline_params)
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.withmpi = self.metadata.options.withmpi
        codeinfo.stdout_name = self._stdout_name

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.retrieve_list = retrieve_list
        calcinfo.retrieve_temporary_list = retrieve_temporary_list
        calcinfo.codes_info = [codeinfo]

        return calcinfo