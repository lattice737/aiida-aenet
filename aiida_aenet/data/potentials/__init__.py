from hashlib import md5
from io import StringIO, BytesIO

from aiida.plugins.entry_point import get_entry_point_names, load_entry_point
from aiida.orm import Data


class AenetPotential(Data):
    """Custom AiiDA Data for aenet potential compatibility with aiida-lammps.
    
    The AenetPotential class only has one type, unlike its aiida-lammps sister 
    EmpiricalPotential, but it still loads an appropriate entry point to align
    with the aiida-lammps implementation.

    Parameters
    ----------
    data : dict
        A dictionary with a list of ANN potential file absolute paths under the
        key "file_contents".

    Examples
    --------
    >>> ann_data = {'Ni.nn': <binary>, 'P.nn': <binary>}
    >>> AenetPotential = (data={
            'file_contents': ann_data
            }
        )
    """

    entry_name = "aenet.potentials"
    pot_lines_fname = "potential_lines.txt"

    def __init__(self, data: dict = None, **kwargs):
        super(AenetPotential, self).__init__(**kwargs)
        self.set_data(data)

    def set_data(self, data: dict = None):
        """Set AenetPotential properties."""

        potential_class = self.load_type("lammps.ann")(data or {})

        elements = potential_class.allowed_element_names

        self.set_attribute("potential_type", "lammps.ann")
        self.set_attribute("atom_style", potential_class.atom_style)
        self.set_attribute("default_units", potential_class.default_units)
        self.set_attribute(
            "allowed_element_names",
            sorted(elements) if elements else elements,
        )

        self.set_file_data(
            potential_lines=potential_class.get_input_potential_lines(),
            potential_files=potential_class.get_external_content() or {},
        )

    def set_file_data(self, potential_lines: list, potential_files: dict):
        """Set AenetPotential file data properties."""

        self.set_attribute("md5|input_lines",
                           md5(potential_lines.encode("utf-8")).hexdigest())
        self.put_object_from_filelike(StringIO(potential_lines),
                                      self.pot_lines_fname)

        external_files = []

        for file, data in potential_files.items():

            self.set_attribute(
                "md5|{}".format(file.replace(".", "_")),
                md5(data).hexdigest(),
            )
            self.put_object_from_filelike(BytesIO(data), file, mode='wb')

            external_files.append(file)

        self.set_attribute("external_files", sorted(external_files))

        for file in self.list_object_names():

            if file not in external_files + [self.pot_lines_fname]:
                self.delete_object(file)

    def get_input_lines(self, kind_symbols: list = None) -> str:
        """Return the pair_style & pair_coeff input lines.

        The __kinds__ placeholder will be replaced with a string of 
        element symbols.

        e.g. for elements S & Cr
             
        pair_style      aenet
        pair_coeff      * * v-1 __kinds__ nn __kinds__
        
        pair_style      aenet
        pair_coeff      * * v-1 S Cr nn S Cr
        """

        content = self.get_object_content(self.pot_lines_fname, "r")

        if kind_symbols:
            content = content.replace("__kinds__", " ".join(kind_symbols))

        return content

    def get_external_files(self) -> dict:
        """Return the file names and binary data of the potential files."""
        return self.get_attribute("external_files")

    @classmethod
    def list_types(cls):
        """Return a list of allowed potential types."""
        return get_entry_point_names(cls.entry_name)

    @classmethod
    def load_type(cls, entry_name: str):
        """Return an instance of the passed entry point."""
        return load_entry_point(cls.entry_name, entry_name)

    @property
    def potential_type(self):
        """Return lammps atom style."""
        return self.get_attribute("potential_type")

    @property
    def atom_style(self):
        """Return lammps atom style."""
        return self.get_attribute("atom_style")

    @property
    def default_units(self):
        """Return lammps default units."""
        return self.get_attribute("default_units")

    @property
    def allowed_element_names(self):
        """Return available atomic symbols."""
        return self.get_attribute("allowed_element_names")

    @property
    def epochs(self):
        """Return integral number of epochs."""
        return self.get_attribute("epochs")

    @property
    def train_mae(self):
        """Return training set mean absolute error."""
        return self.get_attribute("train_mae")

    @property
    def train_rmse(self):
        """Return training set root mean square error."""
        return self.get_attribute("train_rmse")

    @property
    def test_mae(self):
        """Return testing set mean absolute error."""
        return self.get_attribute("test_mae")

    @property
    def test_rmse(self):
        """Return testing set root mean square error."""
        return self.get_attribute("test_rmse")

    @epochs.setter
    def epochs(self, value):
        self.set_attribute("epochs", value)

    @train_mae.setter
    def train_mae(self, value):
        self.set_attribute("train_mae", value)

    @train_rmse.setter
    def train_rmse(self, value):
        self.set_attribute("train_rmse", value)

    @test_mae.setter
    def test_mae(self, value):
        self.set_attribute("test_mae", value)

    @test_rmse.setter
    def test_rmse(self, value):
        self.set_attribute("test_rmse", value)