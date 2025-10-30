from typing import Dict, List, Tuple, Union, Optional, Sequence
from dataclasses import dataclass
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import comb
from copy import deepcopy

from elektrum.rate_function import RateFunc
from elektrum.kinetic_model_helpers import make_encoders


@dataclass
class Link:
    """Simple structure used to derive King-Altman diagrams from Wang algebra.

    Attributes
    ----------
    rates : List or Tuple
        Rate function(s) connecting the states
    states : Tuple[str, str]
        (begin_state, end_state) that this link connects
    gid : int
        Global identifier for this link
    """

    rates: Union[List[RateFunc], Tuple[RateFunc, ...]]
    states: Tuple[str, str]
    gid: int


class KineticModel:
    def __init__(self, param_file: Union[str, dict, Path]):
        """Kinetic model for enzymatic reaction.

        Parameters
        ----------
        param_file : str, dict, or Path
            YAML file path or dictionary containing all parameters necessary
            to build kinetic model including sequence, state, rate, and output
            information.

        Raises
        ------
        ValueError
            If model has fewer than 2 states
        FileNotFoundError
            If param_file path doesn't exist
        """
        self.param_file = param_file

        # Load configuration from file or dict
        self.model_params = self._load_config(param_file)

        # Set up basic attributes
        self.title = self.model_params.get("Title", "kinetic_model")
        self.save_str = self._get_save_path(param_file)
        self.template = self.model_params.get("Input", {}).get("template", None)

        # Create encoders for sequence processing
        seq_values = self.model_params.get("Input", {}).get(
            "values", ["A", "G", "T", "C"]
        )
        self.lab_enc, self.one_enc = make_encoders(seq_values)

        # Validate and set states
        self.states = self.model_params["States"]
        if len(self.states) < 2:
            raise ValueError(
                f"Model must have at least 2 states, got {len(self.states)}"
            )

        # Create rate functions
        self.rates = [
            RateFunc(rate, self.template) for rate in self.model_params["Rates"]
        ]
        self.rate_names = [r.name for r in self.rates]

        # Generate kinetic matrices and links
        (
            self.kinetic_mat,  # 'Matrix' with all kinetic rate objects
            self.link_mat,  # 'Matrix' containing link objects
            self.links,  # List of Link objects
        ) = self.generate_matrices()

        self.links.sort(key=lambda x: x.gid)

    def _load_config(self, param_file: Union[str, dict, Path]) -> dict:
        """Load model configuration from file or dictionary.

        Parameters
        ----------
        param_file : str, dict, or Path
            Configuration file path or dictionary

        Returns
        -------
        dict
            Model configuration parameters

        Raises
        ------
        FileNotFoundError
            If file path doesn't exist
        """
        if isinstance(param_file, dict):
            return param_file

        file_path = Path(param_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path) as yf:
            return yaml.safe_load(yf)

    def _get_save_path(self, param_file: Union[str, dict, Path]) -> str:
        """Determine save path for output files.

        Parameters
        ----------
        param_file : str, dict, or Path
            Configuration file path or dictionary

        Returns
        -------
        str
            Full path for saving output files
        """
        if isinstance(param_file, dict):
            return str(Path.cwd() / self.title)

        return str(Path(param_file).parent / self.title)

    def generate_matrices(self):
        """Make adjacency, kinetic, and link matrix to describe the kinetic
        reactions in model and implement the King-Altman method for finding the
        steady state occupancy of the state of the system.

        Returns
        -------
        numpy.ndarray
            Adjacency matrix of states. Binary and symmetric.
            N_states x N_states
        [[list]]
            2D 'matrix' with all kinetic rate objects.
            N_states x N_states
        [[list]]
            2D 'matrix' containing link objects [description]
        [[list]]
            List of Link objects

        Examples
        --------
        TODO: Add unit tests

        """
        n_states = len(self.states)
        tmp_mat = np.zeros((n_states, n_states))
        kin_mat = tmp_mat.tolist()
        link_mat = tmp_mat.tolist()
        links = []
        already_linked = []

        gid = 0  # Global id
        for rate in self.rates:
            begin_st, end_st = rate.state_list  # beginning state and end state
            bs_i, es_i = self.states.index(begin_st), self.states.index(end_st)

            # Save kinetic matrix for checking reactions later on
            # FYI Indexing may seem backwards at first but it is not.
            #     Remember that off diagonal terms are the rates contributing
            #     to the current state.
            kin_mat[es_i][bs_i] = [rate]

            # Created link matrix. This is important for KingAltman method
            if rate.name not in already_linked:
                reverse_rate = False
                for possible_rev_rate in self.rates:
                    # Indices are swapped
                    if (
                        possible_rev_rate.state_list[0] == end_st
                        and possible_rev_rate.state_list[1] == begin_st
                    ):
                        reverse_rate = possible_rev_rate
                        break
                # Make a new link
                if not reverse_rate:
                    link_mat[bs_i][es_i] = Link([rate], (begin_st, end_st), gid)
                    # Why do I need the reverse rate here?
                    link_mat[es_i][bs_i] = link_mat[bs_i][es_i]
                else:
                    link_mat[bs_i][es_i] = Link(
                        (rate, reverse_rate), (begin_st, end_st), gid
                    )
                    link_mat[es_i][bs_i] = link_mat[bs_i][es_i]
                    already_linked += [reverse_rate]
                links += [link_mat[bs_i][es_i]]
                gid += 1

        # Add rates to the diagnol of the kinetic matrix
        # Need to remember to give negative value to the diagnols later
        for j in range(n_states):
            kin_mat[j][j] = []
            for i in range(n_states):
                if kin_mat[i][j] and i != j:
                    kin_mat[j][j] += kin_mat[i][j]

        return kin_mat, link_mat, links

    def generate_ohe_from_seq(
        self, seq: Union[str, Sequence, np.ndarray]
    ) -> np.ndarray:
        """Get an one hot encoded matrix for a sequence

        Parameters
        ----------
        seq : list, str, ndarray
            Array of n different classes to classify as a number

        Examples
        --------
        TODO: Add unit tests
        """
        tmp_seq = deepcopy(seq)
        if isinstance(seq, list):
            tmp_seq = np.array(seq)
        elif isinstance(seq, str):
            tmp_seq = np.array(list(tmp_seq))
        lab_tmp = self.lab_enc.transform(tmp_seq)
        # one hot encode random sequence i keeping original length
        return self.one_enc.transform(lab_tmp.reshape(-1, 1))

    def generate_rate_list_for_seq(self, seq):
        """Get a list of rates given an array of labels

        Parameters
        ----------
        seq : list, str, ndarray
            Array of n different classes to classify as a number

        Examples
        --------
        TODO: Add unit tests
        """
        # one hot encode random sequence i keeping original length
        seq_ohe = self.generate_ohe_from_seq(seq)
        return [rate.get_rate(seq_ohe) for rate in self.rates]

    def get_kinetic_mat_for_seq(self, seq: str):
        """TODO: Add unit tests and documentation

        Parameters
        ----------
        seq : str
            _description_

        Returns
        -------
        _type_
            _description_
        """

        n = len(self.states)
        seq_ohe = self.generate_ohe_from_seq(seq)
        kin_seq_mat = np.zeros((n, n))
        for i, krow in enumerate(self.kinetic_mat):
            for j, rate_list in enumerate(krow):
                if not rate_list:
                    continue
                for rate in rate_list:
                    kin_seq_mat[i, j] += (
                        -rate.get_rate(seq_ohe) if i == j else rate.get_rate(seq_ohe)
                    )
        return np.array(kin_seq_mat)

    def get_activity(self, seq: str):
        kin_seq_mat = self.get_kinetic_mat_for_seq(seq)
        # Find the eigenvalues of matrix. Sort in descending size order
        eigvals = sorted(np.linalg.eigvals(kin_seq_mat).tolist(), reverse=True)
        for e in eigvals:
            # Structure of matrix means all eigenvalues are <= 0
            assert e <= 0.0
            if e:  # Return the largest non-zero eigenvalue
                return e
        raise ValueError("No eigenvalues found?")

    def get_mutated_seqs(
        self,
        npoints: int,
        mut_num: Union[int, List[int], None] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Generate mutated sequences based on the template sequence.

        The first sequence is always the original template (if it exists).
        Subsequent sequences are mutations of the template according to the
        mutation parameters.

        Parameters
        ----------
        npoints : int
            Number of sequences to generate (including the template).
        mut_num : int, list of int, or None, optional
            Number of mutations per sequence:
            - None: Generate completely random sequences
            - int > 0: Fixed number of mutations per sequence
            - int < 0: Mutate all positions except |mut_num| positions
            - list of int: Variable mutations, weighted by binomial distribution
        rng : np.random.Generator, optional
            Random number generator for reproducibility. If None, uses default RNG.

        Returns
        -------
        np.ndarray
            Array of shape (npoints, seq_length) containing sequences.
            First row is the template (if exists), rest are mutations.

        Examples
        --------
        >>> model = KineticModel("config.yaml")  # template = "AAAA"
        >>> seqs = model.get_mutated_seqs(5, mut_num=2)  # 2 mutations each
        >>> seqs.shape
        (5, 4)
        >>> seqs[0]  # Template sequence
        array(['A', 'A', 'A', 'A'])

        >>> # Variable mutations (1, 2, or 3 mutations per sequence)
        >>> seqs = model.get_mutated_seqs(100, mut_num=[1, 2, 3])

        >>> # Mutate all but 2 positions
        >>> seqs = model.get_mutated_seqs(50, mut_num=-2)
        """
        # Initialize RNG if not provided
        if rng is None:
            rng = np.random.default_rng()

        # Get sequence parameters
        seq_length = int(self.model_params["Input"]["seq_length"])
        seq_values = self.model_params["Input"]["values"]

        # Route to appropriate generation method
        if not self.template or mut_num is None:
            return self._generate_random_sequences(npoints, seq_length, seq_values, rng)
        elif isinstance(mut_num, list):
            return self._generate_variable_mutations(npoints, mut_num, seq_values, rng)
        else:
            return self._generate_fixed_mutations(npoints, mut_num, seq_values, rng)

    def _generate_random_sequences(
        self,
        npoints: int,
        seq_length: int,
        seq_values: List[str],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate completely random sequences.

        If a template exists, the first sequence will be the template.
        """
        seq_arr = rng.choice(seq_values, size=(npoints, seq_length))
        if self.template:
            seq_arr[0, :] = np.array(list(self.template))
        return seq_arr

    def _generate_variable_mutations(
        self,
        npoints: int,
        mut_num: List[int],
        seq_values: List[str],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate sequences with variable number of mutations.

        The number of mutations per sequence is drawn from a binomial-weighted
        distribution over the values in mut_num.
        """
        temp_seq = list(self.template)
        seq_arr = np.repeat([temp_seq], npoints, axis=0)

        # Build mutation options dictionary (exclude original nucleotide)
        opt_dict = self._build_mutation_options(seq_values)

        # Calculate binomial probability weights for each mutation count
        prob_weights = np.array([comb(len(temp_seq), k, exact=True) for k in mut_num])
        prob_weights = prob_weights / prob_weights.sum()

        # Generate mutations for each sequence (skip first, which is template)
        for i in range(1, npoints):
            num_mutations = rng.choice(mut_num, p=prob_weights)
            positions = rng.choice(len(temp_seq), num_mutations, replace=False)

            for pos in positions:
                original_nuc = seq_arr[i, pos]
                seq_arr[i, pos] = rng.choice(opt_dict[original_nuc])

        return seq_arr

    def _generate_fixed_mutations(
        self,
        npoints: int,
        mut_num: int,
        seq_values: List[str],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate sequences with a fixed number of mutations.

        Parameters
        ----------
        mut_num : int
            If positive: number of positions to mutate
            If negative: mutate all except |mut_num| positions
        """
        if mut_num == 0:
            raise ValueError("mut_num cannot be 0. Use None for random sequences.")

        temp_seq = list(self.template)
        seq_length = len(temp_seq)
        seq_arr = np.repeat([temp_seq], npoints, axis=0)

        # Convert negative mutation counts to positive
        # e.g., -2 with length 10 means mutate 8 positions
        num_mutations = mut_num if mut_num > 0 else seq_length + mut_num

        if num_mutations < 0 or num_mutations > seq_length:
            raise ValueError(
                f"Invalid mut_num={mut_num}. Must be in range "
                f"[{-seq_length}, {seq_length}] for sequence length {seq_length}."
            )

        # Build mutation options dictionary
        opt_dict = self._build_mutation_options(seq_values)

        # Generate mutations for each sequence (skip first, which is template)
        for i in range(1, npoints):
            positions = rng.choice(seq_length, num_mutations, replace=False)

            for pos in positions:
                original_nuc = seq_arr[i, pos]
                seq_arr[i, pos] = rng.choice(opt_dict[original_nuc])

        return seq_arr

    def _build_mutation_options(self, seq_values: List[str]) -> Dict[str, List[str]]:
        """Build dictionary mapping each nucleotide to possible mutations.

        For each nucleotide, returns list of all other nucleotides.

        Parameters
        ----------
        seq_values : list of str
            All possible nucleotide values (e.g., ['A', 'T', 'G', 'C'])

        Returns
        -------
        dict
            Maps each nucleotide to list of alternatives.
            e.g., {'A': ['T', 'G', 'C'], 'T': ['A', 'G', 'C'], ...}
        """
        return {nuc: [v for v in seq_values if v != nuc] for nuc in seq_values}

    def gen_simulated_data(self, npoints=1000, mut_num=None, pheno_map=None, **kwargs):
        """Generate data in the form of a .csv to train a neural network to predict kinetics based off a sequence.

        Parameters
        ----------
        npoints : int, optional
            The number of data points to create, by default 1000
        mutation_num : int, optional
            The number of mutations to the template string sequence.
            If negative, all nucleotides will be changed except the
            negative number, by default -1
        pheno_map : str, optional
            The function used to change activity to an experimentally
            measurable phenotype, by default None

        Examples
        --------
        TODO: Add unit tests
        """
        seq_arr = self.get_mutated_seqs(npoints, mut_num)

        data_dict = {
            "seq": [],
            "k_{01}": [],
            "k_{10}": [],
            "k_{12}": [],
            "k_{21}": [],
            "k_{23}": [],
            "k_{32}": [],
            "k_{30}": [],
            "first_eigval": [],
        }

        for i, seq in enumerate(seq_arr):
            data_dict["seq"] += ["".join(seq.tolist())]
            seq_ohe = self.generate_ohe_from_seq(seq)
            for rate in self.rates:
                data_dict[rate.name] += [rate.get_rate(seq_ohe)]

            data_dict["first_eigval"] += [self.get_activity(seq)]

        df = pd.DataFrame.from_dict(data_dict)
        file_name = Path(self.save_str + (".tsv"))
        df.to_csv(file_name, sep="\t", index=False, float_format="%.5f")
