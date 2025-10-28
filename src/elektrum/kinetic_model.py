import sys
from typing import *
import yaml
import random
from pathlib import Path
import numpy as np
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from scipy.special import comb
from copy import deepcopy

from elektrum.kinetic_model_helpers import (
    gen_pos_weight_mat,
    nuc_distr,
    free_energy_mat,
    sigmoid,
    make_encoders,
    single_free_energy_mat,
    single_free_energy_mat_from_kinetic_rates,
)


def convert_nn_rate_to_rate_dict(layer_attrs: dict) -> dict:
    src_st, trg_st = layer_attrs["SOURCE"], layer_attrs["TARGET"]
    input_range = [
        layer_attrs["RANGE_ST"],
        layer_attrs["RANGE_ST"] + layer_attrs["RANGE_D"],
    ]
    ks = layer_attrs["kernel_size"]
    rate_name = "k_{}{}".format(src_st, trg_st)
    rate_dict = {
        "name": rate_name,
        "state_list": [src_st, trg_st],
        "input_range": input_range,
        "kernel_size": ks,
    }
    rate_dict.update(**layer_attrs)
    return rate_dict


def modelSpace_to_modelParams(model_arcs):
    """Convert a neural network 'model space' to kinetic model parameters.

    Example yaml config file for model:

    States: [ '0', '1', '2', '3' ]
    Rates:
    - name: "k_{01}"
        state_list: ['0', '1']
        input_range: [5,10]

    - name: "k_{10}"
        state_list: ['1', '0']
        input_range: [10,15]
    Data:
    - contrib_rate_names: ['k_{30}']
    """
    kinetic_model_params = {
        "States": set([]),
        "Rates": [],
        "Data": {"contrib_rate_names": []},
    }
    states = sorted(
        set(
            [
                s
                for x in model_arcs
                for s in (x.Layer_attributes["SOURCE"], x.Layer_attributes["TARGET"])
            ]
        )
    )
    assert states
    # Create lookup table to place kinetic rates in a sparse matrix.
    # See comment below definition for 'scatter_nd' variable
    scatter_nd_lookup = {s: i for i, s in enumerate(states)}
    for arc in model_arcs:
        if not arc.Layer_attributes.get("EDGE", True):
            continue
        rate_dict = convert_nn_rate_to_rate_dict(arc.Layer_attributes)

        # Update the state list
        src_st, trg_st = rate_dict["state_list"]
        kinetic_model_params["States"].add(src_st)
        kinetic_model_params["States"].add(trg_st)

        # Scatter a flattened kinetic matrix to a sparse matrix as specified by indices.
        # scatter_nd: source is draining, source->target is increasing
        # For more on scatter_nd see https://www.tensorflow.org/api_docs/python/tf/scatter_nd
        scatter_nd = [
            ((scatter_nd_lookup[src_st], scatter_nd_lookup[src_st]), -1),
            ((scatter_nd_lookup[trg_st], scatter_nd_lookup[src_st]), +1),
        ]
        rate_dict["scatter_nd"] = scatter_nd

        # Update the rate list
        kinetic_model_params["Rates"] += [rate_dict]

        # Update the activity contribution list
        if arc.Layer_attributes.get("CONTRIB", False):
            kinetic_model_params["Data"]["contrib_rate_names"].append(rate_dict["name"])

    # Clean up state list
    kinetic_model_params["States"] = sorted(list(kinetic_model_params["States"]))
    return kinetic_model_params


def modelParams_to_modelSpace(model_params):
    """Convert parameters from a kinetic model to a neural network model space."""
    scatter_nd_lookup = {s: i for i, s in enumerate(model_params["States"])}
    for rate in model_params["Rates"]:
        rate["kernel_size"] = 1
        rate["RANGE_ST"] = rate["input_range"][0]
        rate["RANGE_D"] = rate["input_range"][1] - rate["input_range"][0]
        s, t = rate["state_list"]
        # scatter_nd: source is draining, source->target is increasing
        scatter_nd = [
            ((scatter_nd_lookup[s], scatter_nd_lookup[s]), -1),
            ((scatter_nd_lookup[t], scatter_nd_lookup[s]), +1),
        ]
        rate["scatter_nd"] = scatter_nd
    return model_params


class RateFunc:
    """Position weight matrix for a rate"""

    def __init__(self, params: Dict, template: str):
        """
        Initialize a transition rate as a funciton of template sequence given

        Parameters
        ----------
        params : Dict
            _description_
        template : List, optional
            _description_, by default None

        """

        # TODO Make this more readable so users know what attributes RateFunc has
        if "kernel_size" in params:  # TODO Cludge for if we are in modelSpace
            self.__dict__ = convert_nn_rate_to_rate_dict(params)
            self.is_nn_rate = True
        else:
            self.__dict__ = params
            self.is_nn_rate = False
        if "stat_barrier" not in self.__dict__:
            self.stat_barrier = 0
        if "base_rate" not in self.__dict__:
            self.base_rate = 1
        self.template = template
        self.energy_mat = self.build_energy_mat()

    def build_energy_mat(self):
        """TODO: Add unit tests

        Returns
        -------
        _type_
            _description_
        """
        # Length will be used in the weight_distr function
        length = self.input_range[1] - self.input_range[0]
        if self.is_nn_rate:
            # TODO This is not the correct position weight matrix
            return np.zeros((length, 4))
        return eval(self.weight_distr)

    def get_log_rate_vec(self, seq):
        """Equivalent of getting the free energy differences of each nucleotide.
        TODO: Add unit tests
        """
        return -(self.stat_barrier + np.einsum("ij,ij->i", seq, self.energy_mat))

    def get_log_rate(self, seq):
        """TODO: Add unit tests and documentation

        Parameters
        ----------
        seq : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        bi, ei = self.input_range
        return np.log(self.base_rate) - (
            self.stat_barrier + np.einsum("ij,ij", seq[bi:ei], self.energy_mat)
        )

    def get_rate(self, seq):
        """TODO: Add unit tests and documentation

        Parameters
        ----------
        seq : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        bi, ei = self.input_range
        return self.base_rate * np.exp(
            -(self.stat_barrier + np.einsum("ij,ij", seq[bi:ei], self.energy_mat))
        )


class Link:
    """Simple structure used to derive King-Altman diagrams from Wang algebra"""

    def __init__(self, rates, states, gid):
        self.rates = rates
        self.states = states
        self.gid = gid  # Global id


class KineticModel:
    def __init__(self, param_file: Union[str, dict, Path]):
        """Kinetic model for enzymatic reaction.

        Parameters
        ----------
        param_file : str | dict
            Yaml file that contains all parameters necessary to build kinetic model
            including sequence, state, rate, and output information.

        """
        # TODO add flag for neural initialization
        # TODO Check to make sure kinetic model fails if there is only one state
        # TODO Check to make sure that the model is fully connected

        self.param_file = param_file
        # TODO remove this section when we update neural network models
        if isinstance(param_file, dict):
            self.model_params = param_file
            self.title = "kinetic_model"
            self.save_str = str(Path.cwd() / self.title)
            self.template = None
            self.lab_enc, self.one_enc = make_encoders(["A", "G", "T", "C"])

        else:
            with open(Path(param_file)) as yf:
                self.model_params = yaml.safe_load(yf)
            self.title = self.model_params["Title"]
            self.save_str = str(Path(param_file).parent / self.title)
            self.template = self.model_params["Input"].get("template", None)
            self.lab_enc, self.one_enc = make_encoders(
                self.model_params["Input"]["values"]
            )
        # States the system can exist in
        self.states = self.model_params["States"]
        assert len(self.states) > 1

        # Sequence that results in the fastest catalyst rate
        self.rates = [
            RateFunc(rate, self.template) for rate in self.model_params["Rates"]
        ]
        self.rate_names = [r.name for r in self.rates]

        (
            self.kinetic_mat,  # 'Matrix' with all kinetic rate objects
            self.link_mat,  # 'Matrix' containing link objects
            self.links,
        ) = self.generate_matrices()

        self.links.sort(key=lambda x: x.gid)

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
        mut_num: int = None,
        rng: Union[np.random.Generator, None] = None,
    ):
        """Create a list of mutated sequences with the first sequence always
        being the unmutated sequence.

        Parameters
        ----------
        npoints : int
            number of sequences generated
        mut_num : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Examples
        --------
        TODO: Add unit tests
        """
        if not rng:
            rng = np.random.default_rng()
        seq_length = int(self.model_params["Input"]["seq_length"])
        seq_values = self.model_params["Input"]["values"]
        if not self.template or mut_num is None:
            # Randomly generate array of sequences based off input parameters
            seq_arr = rng.choice(seq_values, size=(npoints, seq_length))
            seq_arr[0, :] = np.array(list(self.template))
        elif isinstance(mut_num, list):  # Vary the number of mutations per seq
            temp_seq = list(self.template)
            # 1. Generate all sequences to mutate
            seq_arr = np.repeat([temp_seq], npoints, axis=0)
            opt_dict = {}
            for key in seq_values:
                opt_dict[key] = [v for v in seq_values if v != key]
            # 2. Prep for random mutations
            prob_weights = np.array([comb(len(temp_seq), k) for k in mut_num])
            prob_weights /= prob_weights.sum()

            for i in range(1, npoints):
                ind_choice = rng.choice(
                    len(temp_seq), rng.choice(mut_num, p=prob_weights), replace=False
                )
                for mut_ind in ind_choice:
                    mut_label = seq_arr[i, mut_ind]
                    seq_arr[i, mut_ind] = rng.choice(opt_dict[mut_label])

        else:  # Mutate given template
            assert mut_num != 0
            temp_seq = list(self.template)
            # 1. Generate all sequences to mutate
            seq_arr = np.repeat([temp_seq], npoints, axis=0)
            # TODO check this
            mut_num = mut_num if mut_num > 0 else len(temp_seq) + mut_num

            # 2. Prep for random mutations
            # Make a dictionary of lists with one of the label values removed
            # so that you can randomly choose from the correct list later when
            # making mutations
            opt_dict = {}
            for key in seq_values:
                opt_dict[key] = [v for v in seq_values if v != key]

            # 3. Mutate sequences
            # Choose all indices to mutate for each sequence
            for i in range(1, npoints):
                ind_choice = np.random.choice(len(temp_seq), mut_num, replace=False)
                for mut_ind in ind_choice:
                    mut_label = seq_arr[i, mut_ind]
                    seq_arr[i, mut_ind] = np.random.choice(opt_dict[mut_label])

        return seq_arr

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
