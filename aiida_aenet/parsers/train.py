from io import StringIO
from typing import Dict
import numpy as np

from aiida.parsers.parser import Parser
from aiida.common import exceptions

from aiida_aenet.calculations.train import AenetTrainCalculation
from aiida_aenet.data import potentials
from aiida_aenet.data.potentials import AenetPotential


class AenetTrainParser(Parser):
    """A Parser for AenetTrainCalculation.

    TODO implement logger file parsing
    
    Parameters
    ----------
    node : ProcessNode
        The AenetTrainCalculation node.
    """
    def __init__(self, node):

        super().__init__(node)

        if not issubclass(node.process_class, AenetTrainCalculation):
            raise exceptions.ParsingError(
                "Can only parse AenetTrainCalculation")

    def parse(self, **kwargs):
        """Parse output data."""

        # TODO remote folder options
        # temporary_folder = kwargs["retrieved_temporary_folder"]
        # output_filename = self.node.get_option("output_filename")
        # files_retrieved = self.retrieved.list_object_names()

        results = self.parse_output("train.out")
        potential = self.parse_potential_data(results)

        self.out("potential", potential)  # FIXME nan/inf exception

    def parse_output(self, output_file: str) -> Dict[str, np.ndarray]:
        """Parse output text file for AenetTrainCalculation."""

        with self.retrieved.open(output_file, "r") as handle:
            lines = handle.readlines()

        for i, line in enumerate(lines):

            if f" epoch {11 * ' '} MAE {8 * ' '} <RMSE>" in line:
                break

        n_epochs = self.node.inputs.algorithm.epochs
        nth_epoch_line = min(i + n_epochs + 2, len(lines))
        epoch_lines = lines[i + 1:nth_epoch_line]
        converge_string = " The optimization has converged. Training stopped.\n"

        if converge_string in epoch_lines:

            string_index = epoch_lines.index(converge_string)
            epoch_lines = epoch_lines[:string_index]

        file_stream = StringIO("\n".join(epoch_lines))

        dtype_dict = {
            'names': (
                'epoch',
                'train-mae',
                'train-rmse',
                'test-mae',
                'test-rmse',
                'c',
            ),
            'formats': ('i', 'f', 'f', 'f', 'f', 'S1')
        }

        # FIXME there must be a cleaner solution

        (epochs, train_mae, train_rmse, test_mae, test_rmse,
         _) = np.loadtxt(file_stream, dtype=dtype_dict, unpack=True)

        results = {
            "epochs": epochs,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
        }

        results["epochs"] = results["epochs"][:len(epoch_lines)]

        for key, array in results.items():

            checked = np.where(np.isnan(array), "NaN", array)
            results[key] = list(checked)

        return results

    def parse_potential_data(self, output_data: dict) -> AenetPotential:
        """Set potential metrics to AenetPotential properties."""

        ann_data = {'file_contents': {}}

        for element in self.node.inputs.algorithm.elements:

            binary_data = self.retrieved.open(f"{element}.nn", "rb").read()
            ann_data['file_contents'][f"{element}.nn"] = binary_data

        potential = AenetPotential(data=ann_data)

        # TODO map epochs to mae & rmse data in dicts?
        # FIXME list elements: str -> int/float types

        potential.epochs = output_data["epochs"]
        potential.train_mae = output_data["train_mae"]
        potential.train_rmse = output_data["train_rmse"]
        potential.test_mae = output_data["test_mae"]
        potential.test_rmse = output_data["test_rmse"]

        return potential