"""Rate function module for kinetic models.

This module defines the RateFunc class, which represents a transition rate
between states in a kinetic model. The rate depends on the input sequence
through a position weight matrix (energy matrix).
"""

from typing import Dict, List, Tuple
import numpy as np

from elektrum.kinetic_model_helpers import (
    gen_pos_weight_mat,
    nuc_distr,
    free_energy_mat,
    sigmoid,
    make_encoders,
    single_free_energy_mat,
    single_free_energy_mat_from_kinetic_rates,
)
from elektrum.model_space_utils import convert_nn_rate_to_rate_dict


class RateFunc:
    """Position weight matrix for a transition rate in a kinetic model.

    A rate function calculates the rate of transition between states based on
    the input sequence. The rate depends on a position weight matrix (energy_mat)
    that is calculated from the weight distribution.

    Attributes
    ----------
    name : str
        Rate identifier (e.g., "k_01")
    state_list : List[str]
        States this rate connects [begin_state, end_state]
    input_range : Tuple[int, int]
        Sequence region [start, end] this rate depends on
    weight_distr : str
        Energy matrix calculation expression
    stat_barrier : float
        Static energy barrier added to rate calculation
    base_rate : float
        Base rate constant multiplier
    is_nn_rate : bool
        Whether this is a neural network rate
    template : str
        Template sequence
    energy_mat : np.ndarray
        Position weight matrix of shape (seq_length, 4)
    """

    # Class-level type hints for documentation and type checkers
    name: str
    state_list: List[str]
    input_range: Tuple[int, int]
    weight_distr: str
    stat_barrier: float
    base_rate: float
    is_nn_rate: bool
    template: str
    energy_mat: np.ndarray

    def __init__(self, params: Dict, template: str):
        """Initialize a transition rate function for a kinetic model.

        Parameters
        ----------
        params : Dict
            Rate function parameters containing:
            - name : str
                Name/identifier for this rate (e.g., "k_01")
            - state_list : List[str]
                [begin_state, end_state] this rate connects
            - input_range : Tuple[int, int]
                [start, end] indices of sequence to consider
            - weight_distr : str
                Expression to calculate energy matrix (evaluated)
            - stat_barrier : float, optional
                Static energy barrier (default: 0.0)
            - base_rate : float, optional
                Base rate constant (default: 1.0)
            - kernel_size : int, optional
                If present, indicates neural network rate (for NN models)
        template : str
            Template sequence used for calculating energy matrix

        Raises
        ------
        KeyError
            If required parameters (name, state_list, input_range, weight_distr)
            are missing from params dict.
        """
        # Determine if this is a neural network rate or standard rate
        self.is_nn_rate: bool = "kernel_size" in params

        if self.is_nn_rate:
            # Convert neural network parameters to standard rate dictionary
            params = convert_nn_rate_to_rate_dict(params)

        # Required parameters - fail fast if missing
        self.name: str = params["name"]
        self.state_list: List[str] = params["state_list"]
        self.input_range: Tuple[int, int] = tuple(params["input_range"])
        self.weight_distr: str = params["weight_distr"]

        # Optional parameters with defaults
        self.stat_barrier: float = params.get("stat_barrier", 0.0)
        self.base_rate: float = params.get("base_rate", 1.0)

        # Template and derived attributes
        self.template: str = template
        self.energy_mat: np.ndarray = self.build_energy_mat()

    def __repr__(self) -> str:
        """Return a readable string representation of the rate function.

        Returns
        -------
        str
            String representation showing key attributes
        """
        return (
            f"RateFunc(name={self.name!r}, "
            f"transition={self.state_list[0]}->{self.state_list[1]}, "
            f"base_rate={self.base_rate:.3f}, "
            f"barrier={self.stat_barrier:.3f})"
        )

    def __str__(self) -> str:
        """Return a human-readable string description.

        Returns
        -------
        str
            Detailed string description of the rate function
        """
        return (
            f"Rate {self.name}: {self.state_list[0]} → {self.state_list[1]}\n"
            f"  Base rate: {self.base_rate:.3f}\n"
            f"  Barrier: {self.stat_barrier:.3f}\n"
            f"  Input range: {self.input_range}\n"
            f"  NN rate: {self.is_nn_rate}"
        )

    def __eq__(self, other) -> bool:
        """Compare two rate functions for equality.

        Two rate functions are equal if they have the same name, state_list,
        base_rate, and stat_barrier.

        Parameters
        ----------
        other : RateFunc
            Another rate function to compare with

        Returns
        -------
        bool
            True if rate functions are equal, False otherwise
        """
        if not isinstance(other, RateFunc):
            return NotImplemented
        return (
            self.name == other.name
            and self.state_list == other.state_list
            and np.isclose(self.base_rate, other.base_rate)
            and np.isclose(self.stat_barrier, other.stat_barrier)
        )

    def __hash__(self) -> int:
        """Return hash of rate function for use in sets/dicts.

        Returns
        -------
        int
            Hash value based on name and state_list
        """
        return hash((self.name, tuple(self.state_list)))

    def to_dict(self) -> Dict:
        """Convert rate function to dictionary representation.

        Useful for serialization and debugging.

        Returns
        -------
        Dict
            Dictionary containing all rate function parameters
        """
        return {
            "name": self.name,
            "state_list": self.state_list,
            "input_range": self.input_range,
            "weight_distr": self.weight_distr,
            "stat_barrier": self.stat_barrier,
            "base_rate": self.base_rate,
            "is_nn_rate": self.is_nn_rate,
        }

    def build_energy_mat(self) -> np.ndarray:
        """Build position weight matrix from weight distribution expression.

        The weight_distr string is evaluated as Python code in a controlled
        environment with access to helper functions and numpy.

        Returns
        -------
        np.ndarray
            Position weight matrix of shape (seq_length, 4)

        Examples
        --------
        >>> rate = RateFunc({
        ...     "name": "k_01",
        ...     "state_list": ["E", "ES"],
        ...     "input_range": [0, 5],
        ...     "weight_distr": "free_energy_mat(length, 0.5)"
        ... }, template="AAAAA")
        >>> rate.energy_mat.shape
        (5, 4)
        """
        # Length will be used in the weight_distr function
        length = self.input_range[1] - self.input_range[0]

        if self.is_nn_rate:
            # TODO: This is not the correct position weight matrix
            return np.zeros((length, 4))

        # Define allowed functions and modules for eval
        allowed_globals = {
            "np": np,
            "gen_pos_weight_mat": gen_pos_weight_mat,
            "nuc_distr": nuc_distr,
            "free_energy_mat": free_energy_mat,
            "sigmoid": sigmoid,
            "make_encoders": make_encoders,
            "single_free_energy_mat": single_free_energy_mat,
            "single_free_energy_mat_from_kinetic_rates": single_free_energy_mat_from_kinetic_rates,
        }

        return eval(
            self.weight_distr, allowed_globals, {"self": self, "length": length}
        )

    def get_log_rate_vec(self, seq: np.ndarray) -> np.ndarray:
        """Calculate log rate for each position in the sequence.

        Equivalent to getting the free energy differences of each nucleotide.

        Parameters
        ----------
        seq : np.ndarray
            One-hot encoded sequence of shape (seq_length, 4)

        Returns
        -------
        np.ndarray
            Log rate for each position, shape (seq_length,)

        Examples
        --------
        >>> rate = RateFunc(params, template="AAAA")
        >>> seq_ohe = model.generate_ohe_from_seq("AAAA")
        >>> log_rates = rate.get_log_rate_vec(seq_ohe)
        """
        return -(self.stat_barrier + np.einsum("ij,ij->i", seq, self.energy_mat))

    def get_log_rate(self, seq: np.ndarray) -> float:
        """Calculate total log rate for a sequence.

        Parameters
        ----------
        seq : np.ndarray
            One-hot encoded sequence of shape (seq_length, 4)

        Returns
        -------
        float
            Total log rate for the sequence

        Examples
        --------
        >>> rate = RateFunc(params, template="AAAA")
        >>> seq_ohe = model.generate_ohe_from_seq("AAAA")
        >>> log_rate = rate.get_log_rate(seq_ohe)
        """
        bi, ei = self.input_range
        return np.log(self.base_rate) - (
            self.stat_barrier + np.einsum("ij,ij", seq[bi:ei], self.energy_mat)
        )

    def get_rate(self, seq: np.ndarray) -> float:
        """Calculate transition rate for a sequence.

        This is the main method for computing the rate of transition based
        on the sequence. The rate is calculated as:

        rate = base_rate * exp(-(stat_barrier + sequence_energy))

        Parameters
        ----------
        seq : np.ndarray
            One-hot encoded sequence of shape (seq_length, 4)

        Returns
        -------
        float
            Transition rate for the sequence

        Examples
        --------
        >>> rate = RateFunc(params, template="AAAA")
        >>> seq_ohe = model.generate_ohe_from_seq("AAAA")
        >>> transition_rate = rate.get_rate(seq_ohe)
        >>> transition_rate > 0
        True
        """
        bi, ei = self.input_range
        return self.base_rate * np.exp(
            -(self.stat_barrier + np.einsum("ij,ij", seq[bi:ei], self.energy_mat))
        )
