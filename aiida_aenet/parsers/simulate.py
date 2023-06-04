from aiida_lammps.parsers.lammps.md_multi import MdMultiParser


class AenetLammpsMdParser(MdMultiParser):
    """Parser for aenet-lammps AenetLammpsMdCalculations."""
    def __init__(self, node):
        """Initialize the instance of Lammps MD Parser."""
        super(AenetLammpsMdParser, self).__init__(node)