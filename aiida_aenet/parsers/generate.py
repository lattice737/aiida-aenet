# -*- coding: utf-8 -*-
from aiida.orm import SinglefileData
from aiida.parsers.parser import Parser
from aiida.common import exceptions

from aiida_aenet.calculations.generate import AenetGenerateCalculation


class AenetGenerateParser(Parser):
    """A Parser for AenetGenerateCalculation.

    TODO implement stdout & logger file parsing
    
    Parameters
    ----------
    node : ProcessNode
        The AenetGenerateCalculation node. 
    """
    def __init__(self, node):

        super().__init__(node)

        if not issubclass(node.process_class, AenetGenerateCalculation):
            raise exceptions.ParsingError(
                "Only parses AenetGenerateCalculation")

    def parse(self, **kwargs):
        """Parse train file to SinglefileData."""

        # REMOTE FOLDER OPTIONS
        # temporary_folder = kwargs["retrieved_temporary_folder"]
        # output_filename = self.node.get_option("output_filename")
        # files_retrieved = self.retrieved.list_object_names()

        train_file = "train.dat"

        with self.retrieved.open(train_file, "rb") as handle:
            file_node = SinglefileData(file=handle)

        self.out("train_file", file_node)