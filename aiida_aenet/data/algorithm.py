from typing import Union

from aiida.orm import Data

from .potentials import AenetElement


class AenetAlgorithm(Data):
    """A custom AiiDA class for preserving the state of an aenet neural network.
    
    Parameters
    ----------
    elements : dict
        A dictionary of data for each unique element in the reference set,
        including atomic energy (eV), number of nodes, and the network
        architecture of each layer.
    descriptor : dict
        A dictionary of descriptor data, including the descriptor type and
        its parameters.
    training : str | dict
        A dictionary of the training method data (or the string 'bfgs'),
        including the training method type and its parameters.
    parameters : dict
        A dictionary of the neural network algorithm learning parameters and
        logging settings.

    Examples
    --------
    >>> elements = {
            "Ni": {
                "energy":
                -4676.3936784796315,
                "nodes":
                2,
                "network": [{
                    "nodes": 2,
                    "activation": "tanh"
                }, {
                    "nodes": 2,
                    "activation": "tanh"
                }]
            },
            "P": {
                "energy":
                -194.14828627169916,
                "nodes":
                2,
                "network": [{
                    "nodes": 2,
                    "activation": "tanh"
                }, {
                    "nodes": 2,
                    "activation": "tanh"
                }]
            }
        }
    >>> chebyshev = {
            "type": "chebyshev",
            "parameters": {
                "radial_rc": 4.0,
                "radial_n": 6,
                "angular_rc": 4.0,
                "angular_n": 2
            }
        }
    >>> parameters = {
            "test_percent": 10,
            "epochs": 10,
            "max_energy": 0.0,
            "r_min": 0.75
        }
    >>> nn_algorithm = AenetAlgorithm(
            elements=elements,
            descriptor=chebyshev,
            training="bfgs",
            parameters=parameters,
        )
    """

    training_methods = ("lm", "gd")

    # TODO implement number of structures attr?
    # TODO implement logger? some way to tell user which steps this algorithm has been through already (json?)
    # TODO implement input validation; check for other instances instead of dict

    def __init__(
        self,
        elements: dict,
        descriptor: dict,
        training: Union[str, dict],
        parameters: dict,
        **kwargs,
    ):

        super(AenetAlgorithm, self).__init__(**kwargs)

        self.set_attribute("elements", elements)
        self.set_descriptor(descriptor)
        self.set_parameters(parameters)

        if isinstance(training, str) and training == 'bfgs':
            self.set_attribute("train_method", training)
        elif isinstance(training,
                        dict) and training["method"] in self.training_methods:
            self.set_attribute("train_method", training["method"])
            self.set_training(training)
        else:
            raise Exception

    # FIXME AenetElement instances not serializable
    def set_elements(self, elements_dict: dict):
        """Parse AenetElement data to AenetAlgorithm properties."""

        element_list = []

        for symbol, traits in elements_dict.items():

            element = AenetElement(symbol, traits)
            element_list.append(element)

        self.set_attribute("elements", element_list)

    # TODO reimplement when Descriptor class implemented
    def set_descriptor(self, descriptor_dict: dict):
        """Parse AenetDescriptor data to AenetAlgorithm properties."""

        self.set_attribute("descriptor", descriptor_dict)

    def set_training(self, training_dict: dict):
        """Parse AenetTrainMethod data to AenetAlgorithm properties."""

        lm_defaults = (
            ("batchsize", 5000),
            ("learn_rate", 0.1),
            ("rate_adjust", 5.0),
            ("optimize_iterations", 3),
            ("converge_threshold", 0.001),
        )

        gd_defaults = (
            ("learn_rate", 0.003),
            ("momentum_rate", 0.05),
        )

        default_tuple = locals()[f"{training_dict['type']}_defaults"]
        parameters = training_dict["parameters"]

        for attr, default in default_tuple:
            parameters.setdefault(attr, default)

        self.set_attribute("training_parameters", parameters)

    def set_parameters(self, parameter_dict: dict):
        """Set neural-network algorithm parameters to AenetAlgorithm properties."""

        defaults = (
            ("debug", True),
            ("timing", True),
            ("save_energies", True),
            ("predict_forces", False),
            ("predict_relax", False),
            ("test_percent", 10),
            ("epochs", 10),
            ("max_energy", 0.0),
            ("r_min", 0.75),
        )

        for attr, default in defaults:

            value = parameter_dict.setdefault(attr, default)
            self.set_attribute(attr, value)

    @property
    def elements(self):
        """Return elements data dictionary."""
        return self.get_attribute("elements")

    @property
    def descriptor(self):
        """Return descriptor data dictionary."""
        return self.get_attribute("descriptor")

    @property
    def debug(self):
        """Return debugging flag."""
        return self.get_attribute("debug")

    @property
    def timing(self):
        """Return timing flag."""
        return self.get_attribute("timing")

    @property
    def save_energies(self):
        """Return save_energies flag."""
        return self.get_attribute("save_energies")

    @property
    def predict_forces(self):
        """Return forces flag for AenetPredictCalculation."""
        return self.get_attribute("predict_forces")

    @property
    def predict_relax(self):
        """Return relaxation flag for AenetPredictCalculation."""
        return self.get_attribute("predict_relax")

    @property
    def train_method(self):
        """Return training method data dictionary."""
        return self.get_attribute("train_method")

    @property
    def test_percent(self):
        """Return test proportion of reference set."""
        return self.get_attribute("test_percent")

    @property
    def epochs(self):
        """Return number of learning iterations."""
        return self.get_attribute("epochs")

    @property
    def max_energy(self):
        """Return maximum energy value."""
        return self.get_attribute("max_energy")

    @property
    def r_min(self):
        """Return minimum interatomic radius."""
        return self.get_attribute("r_min")
