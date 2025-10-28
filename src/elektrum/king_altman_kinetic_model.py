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

from elektrum.kinetic_model import KineticModel


def wang_algebra_sequences(branch_list: List) -> List:
    """Recursive function to find all KA diagrams using Wang algebra which
    automatically checks all duplicate states.


    Parameters
    ----------
    branch_list : List
        List of lists of all kinetic rates that go to nodes

    Returns
    -------
    List
        Return a list of lists of all acceptable KA patterns checking
    for duplicate states.

    Examples
    --------
    >>> print(wang_algebra_sequences([[2, 5], [5, 6], [7, 2]]))
    [[2, 5, 7], [2, 6, 7], [5, 6, 7], [5, 6, 2]]
    """

    assert len(branch_list) > 0
    if len(branch_list) == 1:
        return [[state] for state in branch_list[0]]

    seq = []
    sub_seq_list = wang_algebra_sequences(branch_list[1:])
    for sseq in sub_seq_list:
        for state in branch_list[0]:
            if state not in sseq:
                seq += [[state] + sseq]
    return seq


class KingAltmanKineticModel(KineticModel):
    def __init__(self, param_file: Union[str, dict]):
        """Kinetic model for a steady state enzymatic reaction.

        Parameters
        ----------
        param_file : str
            Yaml file that contains all parameters necessary to build kinetic model
            including sequence, state, rate, and output information.

        """
        super().__init__(param_file)
        self.build_ka_patterns()

        self.contrib_rate_names = self.model_params["Data"]["contrib_rate_names"]
        self.contrib_rates = []
        self.contrib_occupancy_funcs = []
        for contrib_name in self.contrib_rate_names:
            for rate in self.rates:
                if rate.name == contrib_name:
                    self.contrib_rates += [rate]
                    # Get the beginning state of the contributing rate
                    self.contrib_occupancy_funcs += [
                        self.get_state_occupancy_func(rate.state_list[0])
                    ]

        assert len(self.contrib_rate_names) > 0

    def build_ka_patterns(self):
        """Build a list of state sequences that are used in the King-Altman method."""
        n_states = len(self.states)
        node_branch_list = []
        # Build up node branch list with first state cut from the link matrix
        for i in range(1, n_states):  # skip the first row
            node_branch_list += [[]]
            for j in range(n_states):
                if self.link_mat[i][j] != 0:
                    node_branch_list[-1] += [self.link_mat[i][j].gid]

        self.ka_patterns = wang_algebra_sequences(node_branch_list)

    def get_numerator(self, state):
        """Get numerator of kinetic ratio related for state"""
        rate_prod_list = []
        # Iterate through all patterns and choose the correct direction
        for kap in self.ka_patterns:
            # 1. Find all the links that have rates that end at the state
            # 2. If unused links still exist, find which ones have rates that end at the beginnings
            #    states of already chosen rates. Repeat process until all links are used
            # 2a. If no rates are found, return an empty list for that pattern
            rate_prod_list += [self.calc_KA_rate_product([state], kap)]
        # Return a list of rate lists representing rate products removing any
        # empty lists
        return [r for r in rate_prod_list if r]

    def get_denominator(self):
        """Denominator is just the sum of all the numerators"""
        return sum([self.get_numerator(state) for state in self.states], [])

    def calc_KA_rate_product(self, end_states, link_ids):
        """Recursive function that finds the correct direction for links given a KA pattern sequence.
        If no path exists for that pattern because of irreversibility, an empty
        list is returned instead.
        """
        rate_list = []
        new_end_states = []
        used_link_ids = []
        for l_id in link_ids:
            for es in end_states:
                for rate in self.links[l_id].rates:
                    # If rate ends in an end state, add it to rate list
                    if rate.state_list[1] == es:
                        rate_list += [rate]
                        # Next recursive statement needs to know about new end
                        # state
                        new_end_states += [rate.state_list[0]]
                        used_link_ids += [l_id]
                        break
                # Don't need to continue loop if link was used
                if l_id in used_link_ids:
                    break

        # Find all links not used and pass them to next recursion step
        unused_link_ids = [l_id for l_id in link_ids if l_id not in used_link_ids]
        if len(unused_link_ids) == len(link_ids):
            # Could not find kinetic rate for pattern
            return []
        if len(unused_link_ids) == 0:
            return rate_list
        next_rate = self.calc_KA_rate_product(new_end_states, unused_link_ids)
        if not next_rate:
            return []
        return rate_list + next_rate

    def get_state_occupancy_func(self, state):
        """Create and return a function that calculates the total occupancy of an enzymatic state given a sequence."""
        numer_list = self.get_numerator(state)
        denom_list = self.get_denominator()

        def get_state_occupancy(seq):
            numer = 0.0
            denom = 0.0
            for rlist in numer_list:
                term = 1.0
                for nrate in rlist:
                    term *= nrate.get_rate(seq)
                numer += term

            for dlist in denom_list:
                term = 1.0
                for drate in dlist:
                    term *= drate.get_rate(seq)
                denom += term

            return numer / denom

        return get_state_occupancy

    def get_activity(self, seq: str):
        tmp = self.lab_enc.transform(list(seq))
        # one hot encode random sequence i keeping original length
        seq_ohe = self.one_enc.transform(tmp.reshape(-1, 1))
        # loop over all rates to get associated values for the sequence

        # loop over occupancy functions and associated contrib_rates
        act = 0.0
        for rate, ofunc in zip(self.contrib_rates, self.contrib_occupancy_funcs):
            act += rate.get_rate(seq_ohe) * ofunc(seq_ohe)

        return act

    def get_ka_pattern_mat(self):
        """Return a binary matrix that relates rate vector to a term vector
            t_i = A_{ij} k_j
            ^      ^ --|  ^------------|
        term vector   KA matrix     rate vector
        (e.g [(k_1 * k_2 * k_3), ...)

        TODO: Add unit tests
        """
        denom_list = self.get_denominator()
        ka_mat = np.zeros((len(denom_list), len(self.rates)))
        for i, term in enumerate(denom_list):
            for r in term:
                ka_mat[i, self.rates.index(r)] = 1
        return ka_mat

    def gen_simulated_data(
        self,
        npoints: int = 1000,
        rng_seed: int = 1234,
        # contrib_rate_names: List = None,
        pheno_map: str = None,
        mut_num: int = None,
        **kwargs,
    ):
        """Generate data in the form of a .csv to train a neural network to predict kinetics based off a sequence.

        Parameters
        ----------
        npoints : int, optional
            The number of data points to create, by default 1000
        contrib_rate_names : list, optional
            The rates from the model that contribute to the activity.
            Must not be empty. by default None
        pheno_map : str, optional
            The function used to change activity to an experimentally
            measurable phenotype, by default None
        mutation_num : int, optional
            The number of mutations to the template string sequence.
            If negative, all nucleotides will be changed except the
            negative number, by default -1

        Examples
        --------
        TODO: Add unit tests

        """
        assert self.contrib_rate_names

        # Create occupancy functions for states that contribute to pheno_map
        # based on the list of contrib_rates
        contrib_rates = []
        occupancy_funcs = []
        for contrib_name in self.contrib_rate_names:
            for rate in self.rates:
                if rate.name == contrib_name:
                    contrib_rates += [rate]
                    # Get the beginning state of the contributing rate
                    occupancy_funcs += [
                        self.get_state_occupancy_func(rate.state_list[0])
                    ]

        rng = np.random.default_rng(rng_seed)
        seq_arr = self.get_mutated_seqs(npoints, mut_num, rng)

        rates_arr = np.zeros((npoints, len(self.rates)))
        act_arr = np.zeros((npoints))
        pheno_arr = np.zeros((npoints))
        pheno_map_func = eval(pheno_map) if pheno_map else lambda x: x

        for i, seq in enumerate(seq_arr):
            tmp = self.lab_enc.transform(seq)
            # one hot encode random sequence i keeping original length
            seq_ohe = self.one_enc.transform(tmp.reshape(-1, 1))
            # loop over all rates to get associated values for the sequence
            for j, rate in enumerate(self.rates):
                rates_arr[i, j] = rate.get_rate(seq_ohe)

            # loop over occupancy functions and associated contrib_rates
            for rate, ofunc in zip(contrib_rates, occupancy_funcs):
                # Get occupancy and multiply by contrib_rate value
                # Sum all terms to get pheno_map
                act_arr[i] += rate.get_rate(seq_ohe) * ofunc(seq_ohe)
            pheno_arr[i] = pheno_map_func(act_arr[i])

        # Save phenotype to file
        seq_list = ["".join(seq) for seq in seq_arr.tolist()]
        combined_arr = np.hstack(
            (
                np.array(seq_list).reshape(-1, 1),
                rates_arr,
                act_arr.reshape(-1, 1),
                pheno_arr.reshape(-1, 1),
            )
        )

        df = pd.DataFrame(combined_arr)
        df.columns = (
            ["seq"]
            + [rate.name for rate in self.rates]
            + ["raw_activity"]
            + ["phenotype_activity"]
        )
        # Convert numeric columns to float
        for col in df.columns:
            if col != "seq":  # Skip the "seq" column
                df[col] = pd.to_numeric(df[col], errors="coerce")

        file_name = Path(self.save_str + (".csv"))
        print(df.dtypes)
        df.to_csv(file_name, sep=",", float_format="%.5g")
        self.save_ka_matrix()
        self.save_rate_contrib_matrix(self.contrib_rate_names)

    def save_ka_matrix(self):
        ka_mat = self.get_ka_pattern_mat()
        np.savetxt(Path(self.save_str + "_ka_mat.nptxt"), ka_mat, fmt="%d")

    def save_rate_contrib_matrix(self, contrib_rate_names, save=True):
        """TODO: Swap with get_rate_contrib_matrix"""
        denom_list = self.get_denominator()
        contrib_mat = np.zeros((len(self.rates), len(denom_list)))
        for i, crate in enumerate(self.rates):
            if crate.name in contrib_rate_names:
                start_state = crate.state_list[0]
                nterms = self.get_numerator(start_state)
                for j, kap in enumerate(denom_list):
                    if kap in nterms:
                        contrib_mat[i, j] = 1.0
        if save is True:
            np.savetxt(
                Path(self.save_str + "_rate_contrib_mat.nptxt"), contrib_mat, fmt="%d"
            )
        return contrib_mat

    def get_rate_contrib_matrix(self):
        """TODO: Add documentation and unit tests. Swap with save_rate_contrib_matrix

        Returns
        -------
        _type_
            _description_
        """
        contrib_rate_names = self.model_params["Data"]["contrib_rate_names"]
        return self.save_rate_contrib_matrix(contrib_rate_names, save=False)
