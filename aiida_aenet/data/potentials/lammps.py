from aiida_lammps.data.pot_plugins.base_plugin import PotentialAbstract


class ANN(PotentialAbstract):
    """A class for creation of aenet ANN potential inputs."""
    def validate_data(self, data):
        """Validate the input data."""

        assert "file_contents" in data, data

    def get_external_content(self) -> dict:
        """Return raw file content."""
        return self.data["file_contents"]

    def get_input_potential_lines(self):
        """Return the template pair_style & pair_coeff input lines."""

        version = "v-1"  # TODO options: v00, v-1
        suffix = "nn"  # potential name will be X.suffix

        lammps_input_text = "pair_style      aenet\n"
        lammps_input_text += (
            f"pair_coeff      * * {version} __kinds__ {suffix} __kinds__\n")

        return lammps_input_text

    @property
    def allowed_element_names(self):
        """Return the list of elements with potentials."""
        return self.data.get("element_names", None)

    @property
    def atom_style(self):
        """Return the atom style string."""
        return "atomic"

    @property
    def default_units(self):
        """Return the default units string."""
        return "metal"