from aiida.engine import WorkChain, ToContext
from aiida.orm import List

from aiida_aenet.calculations.generate import AenetGenerateCalculation
from aiida_aenet.calculations.predict import AenetPredictCalculation
from aiida_aenet.calculations.train import AenetTrainCalculation
from aiida_aenet.data.potentials import AenetPotential
from aiida_aenet.data.algorithm import AenetAlgorithm


class MakePotentialWorkChain(WorkChain):
    """
    A WorkChain that runs the three major aenet calculations with aiida-aenet.

    Parameters
    ----------
    algorithm : AenetAlgorithm
        A custom Node containing neural-network algorithm parameters, including
        element & descriptor parameters.
    reference : List
        A List node of PwCalculation PKs for configuration, total energy, and
        atomic force data.
    generate : dict
        The input dictionary of AenetGenerateCalculation.
    train : dict
        The input dictionary of AenetTrainCalculation.
    predict : dict
        The input dictionary of AenetPredictCalculation.

    Returns
    -------
    potential : AenetPotential
        A custom Node compatible with aiida-lammps that contains atomic
        potential data.
    """
    @classmethod
    def define(cls, spec):
        """WorkChain I/O specification & outline."""
        super().define(spec)
        spec.input('algorithm', valid_type=AenetAlgorithm)
        spec.input('reference', valid_type=List)
        spec.expose_inputs(
            AenetGenerateCalculation,
            namespace='generate',
            exclude=('algorithm', 'reference'),
        )
        spec.expose_inputs(
            AenetTrainCalculation,
            namespace='train',
            exclude=('algorithm', 'train_file'),
        )
        spec.expose_inputs(
            AenetPredictCalculation,
            namespace='predict',
            exclude=('algorithm', 'reference', 'potential'),
        )

        spec.outline(
            cls.setup,
            cls.generate,
            cls.train,
            cls.predict,
            cls.get_potential,
        )

        spec.output('potential', valid_type=AenetPotential)

    def setup(self):
        """Assign input dictionaries to context for outline calculations."""

        self.ctx.generate = self.exposed_inputs(
            AenetGenerateCalculation,
            namespace="generate",
        )

        self.ctx.train = self.exposed_inputs(
            AenetTrainCalculation,
            namespace="train",
        )

        self.ctx.predict = self.exposed_inputs(
            AenetPredictCalculation,
            namespace="predict",
        )

    def get_potential(self):
        """Return AenetPotential node as WorkChain output."""

        self.out("potential", self.ctx.train_calc.outputs.potential)

    def generate(self):
        """Submit AenetGenerateCalculation."""

        inputs = {
            "code": self.ctx.generate["code"],
            "reference": self.inputs.reference,
            "algorithm": self.inputs.algorithm,
        }

        generate_x = self.submit(AenetGenerateCalculation, **inputs)

        return ToContext(generate_calc=generate_x)

    def train(self):
        """Submit AenetTrainCalculation."""

        inputs = {
            "code": self.ctx.train["code"],
            "algorithm": self.inputs.algorithm,
            "train_file": self.ctx.generate_calc.outputs.train_file,
        }

        train_x = self.submit(AenetTrainCalculation, **inputs)

        return ToContext(train_calc=train_x)

    def predict(self):
        """Submit AenetPredictCalculation."""

        inputs = {
            "code": self.ctx.predict["code"],
            "reference": self.inputs.reference,
            "algorithm": self.inputs.algorithm,
            "potential": self.ctx.train_calc.outputs.potential,
        }

        predict_x = self.submit(AenetPredictCalculation, **inputs)

        return ToContext(predict_calc=predict_x)
