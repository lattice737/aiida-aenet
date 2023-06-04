import re

from aiida.parsers.parser import Parser
from aiida.common import exceptions
from aiida.orm import Dict

from aiida_aenet.calculations.predict import AenetPredictCalculation


class AenetPredictParser(Parser):
    """A Parser for AenetPredictCalculation.

    TODO implement logger file parsing
    
    Parameters
    ----------
    node : ProcessNode
        The AenetPredictCalculation node.
    """
    def __init__(self, node):

        super().__init__(node)

        if not issubclass(node.process_class, AenetPredictCalculation):
            raise exceptions.ParsingError(
                "Can only parse AenetPredictCalculation")

    def parse(self, **kwargs):
        """Parse output data."""

        # TODO remote folder options
        # temporary_folder = kwargs["retrieved_temporary_folder"]
        # output_filename = self.node.get_option("output_filename")
        # files_retrieved = self.retrieved.list_object_names()

        results = self.parse_output("predict.out")

        self.out("results", Dict(dict=results))

    def parse_output(self, output_file) -> dict:
        """Parse predict.x output to a dictionary."""

        # FIXME files -> pks; file names have no meaning in AiiDA

        pks = [pk for pk in self.node.inputs.reference.get_list()]

        natoms, energies = [], []
        energy_evaluation = False

        with self.retrieved.open(output_file, "r") as f:

            for line in f:

                if not energy_evaluation and 'Energy evaluation' in line:
                    energy_evaluation = True

                if not energy_evaluation:
                    continue

                if 'Number of atoms' in line:
                    natoms.append(int(line.replace(' Number of atoms   :', '')))

                if f" Total energy {13 * ' '} :" in line:
                    m = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                    energies.append(float(m[0]))

                if 'Atomic Energy Network done.' in line:
                    break

        pk_dict = {
            str(pk): {
                "n": n,
                "E": E
            }
            for (pk, n, E) in zip(pks, natoms, energies)
        }

        return pk_dict