from aiida.engine import WorkChain, append_, while_
from aiida.orm import Code, StructureData

from aiida_aenet.calculations.simulate import AenetLammpsMdCalculation
from aiida_aenet.data.potentials import AenetPotential

from aiida_lammps.calculations.lammps.md_multi import MdMultiCalculation
from aiida_lammps.data.potential import EmpiricalPotential


class CompareSimulationsWorkChain(WorkChain):
    """Simulate an empirical and ANN potential with aiida-lammps in parallel.

    FIXME aiida interprets both of the calcs as finished when only first one
    has finished
    FIXME return specifically relevant data as output, rather than the calc nodes?
    
    Parameters
    ----------
    code : Code
        A Code node linked to the external lammps executable.
    structure : StructureData
        The StructureData node to simulate.
    ann_potential : AenetPotential
        A custom Node with the ANN potential file data, suitable for aiida-lammps
        and an aenet-lammps build of LAMMPS.
    empirical_potential : EmpiricalPotential
        The aiida-lammps potential object.
    md : dict
        The input dictionary of MdMultiCalculation (aiida-lammps).
    
    Returns
    -------
    empirical_simulation : MdMultiCalculation
        The aiida-lammps CalcJob node with empirical potential input.
    ann_simulation : MdMultiCalculation
        The aiida-lammps CalcJob node with ANN potential input.
    """
    @classmethod
    def define(cls, spec):
        """WorkChain I/O specification and outline."""
        super().define(spec)
        spec.input('code', valid_type=Code)
        spec.input('structure', valid_type=StructureData)
        spec.input('ann_potential', valid_type=AenetPotential)
        spec.input('empirical_potential', valid_type=EmpiricalPotential)
        spec.expose_inputs(
            MdMultiCalculation,
            namespace="simulation",
            exclude=('code', 'structure', 'potential'),
        )

        spec.outline(
            cls.setup,
            cls.simulate,
            cls.log,
        )

    def setup(self):
        """Assign input dictionary to context."""

        self.ctx.simulation_inputs = self.exposed_inputs(
            MdMultiCalculation,
            namespace="simulation",
        )

        self.ctx.simulation_inputs["code"] = self.inputs.code
        self.ctx.simulation_inputs["structure"] = self.inputs.structure

    def simulate(self):
        """Submit MdMultiCalculation jobs with input potentials."""

        simulations = (MdMultiCalculation, AenetLammpsMdCalculation)
        potentials = (
            self.inputs.empirical_potential,
            self.inputs.ann_potential,
        )

        for V, MD in zip(potentials, simulations):

            self.ctx.simulation_inputs["potential"] = V
            simulation = self.submit(MD, **self.ctx.simulation_inputs)
            self.to_context(simulations=append_(simulation))

    def log(self):
        """Log the WorkChain completion string."""
        self.report("The WorkChain has finished.")
