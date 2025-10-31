from typing import *  # noqa: F403
from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd
import h5py
import yaml

from .kinetic_model import KineticModel


def wang_algebra_sequences(branch_list):
    """Return a list of lists of all acceptable KA patterns checking
    for duplicate states.

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
    """King-Altman kinetic model for steady-state enzymatic reactions.

    This class focuses on King-Altman pattern construction and occupancy
    calculations. It delegates configuration, sequence handling and rate
    construction to the base `KineticModel`.
    """

    def __init__(self, param_file: Union[str, dict]):
        super().__init__(param_file)
        # Build KA patterns and precompute contribution lookups
        self.build_ka_patterns()

        self.contrib_rate_names = self.model_params["Data"].get(
            "contrib_rate_names", []
        )
        assert self.contrib_rate_names, "Must specify contribution rate names"

        # Precompute contributing rates and occupancy functions
        self.contrib_rates = [
            r for r in self.rates if r.name in self.contrib_rate_names
        ]
        self.contrib_occupancy_funcs = [
            self.get_state_occupancy_func(r.state_list[0]) for r in self.contrib_rates
        ]

    def build_ka_patterns(self) -> None:
        """Create King-Altman patterns from the link matrix."""
        n_states = len(self.states)
        node_branch_list: List[List[int]] = []
        for i in range(1, n_states):
            node_branch_list.append(
                [
                    self.link_mat[i][j].gid
                    for j in range(n_states)
                    if self.link_mat[i][j] != 0
                ]
            )
        self.ka_patterns = wang_algebra_sequences(node_branch_list)

    def get_numerator(self, state: str) -> List[List]:
        """Return list of rate-product lists for a given state."""
        products = [self.calc_KA_rate_product([state], kap) for kap in self.ka_patterns]
        return [p for p in products if p]

    def calc_KA_rate_product(self, end_states: List[str], link_ids: List[int]) -> List:
        """Recursive helper to compute a single KA pattern's rate-product list."""
        if not link_ids:
            return []

        used_rates = []
        new_end_states = []
        used_link_ids = []

        for l_id in link_ids:
            for es in end_states:
                for rate in self.links[l_id].rates:
                    if rate.state_list[1] == es:
                        used_rates.append(rate)
                        new_end_states.append(rate.state_list[0])
                        used_link_ids.append(l_id)
                        break
                if l_id in used_link_ids:
                    break

        unused = [lid for lid in link_ids if lid not in used_link_ids]
        if len(unused) == len(link_ids):
            return []
        if not unused:
            return used_rates

        next_rates = self.calc_KA_rate_product(new_end_states, unused)
        if not next_rates:
            return []
        return used_rates + next_rates

    def get_state_occupancy_func(self, state: str):
        """Return a function that computes occupancy for `state` given seq OHE."""
        numerators = self.get_numerator(state)
        denominators = self.get_denominator()

        def occupancy(seq_ohe):
            num = 0.0
            den = 0.0
            for ka_term in numerators:
                prod = 1.0
                for r in ka_term:
                    prod *= r.get_rate(seq_ohe)
                num += prod
            for ka_term in denominators:
                prod = 1.0
                for r in ka_term:
                    prod *= r.get_rate(seq_ohe)
                den += prod
            return num / den

        return occupancy

    def get_denominator(self) -> List[List]:
        """Return denominator terms (sum of numerators across states)."""
        return sum((self.get_numerator(s) for s in self.states), [])

    def get_activity(self, seq: str) -> float:
        """Compute activity for a raw sequence string using precomputed contributors."""
        # Use base class encoders
        seq_ohe = self.generate_ohe_from_seq(seq)
        activity = 0.0
        for rate, occ in zip(self.contrib_rates, self.contrib_occupancy_funcs):
            activity += rate.get_rate(seq_ohe) * occ(seq_ohe)
        return activity

    def get_eigen_activity(self, seq: str) -> float:
        """Compute activity for a one-hot encoded sequence using precomputed contributors."""
        eigval, eigvec = np.linalg.eig(self.get_kinetic_mat_for_seq(seq))
        steady_state_index = np.argmin(np.abs(eigval))
        steady_state_vec = eigvec[:, steady_state_index]
        steady_state_vec = steady_state_vec / np.sum(steady_state_vec)
        state_contrib = {r.state_list[0]: r for r in self.contrib_rates}
        seq_ohe = self.generate_ohe_from_seq(seq)
        activity = 0.0
        for i, state in enumerate(self.states):
            rate = state_contrib.get(state, None)
            if rate:
                activity += steady_state_vec[i] * rate.get_rate(seq_ohe)
        return activity

    def get_ka_pattern_mat(self) -> np.ndarray:
        """Return a binary matrix mapping KA terms to rates."""
        denom_list = self.get_denominator()
        ka_mat = np.zeros((len(denom_list), len(self.rates)), dtype=int)
        for i, term in enumerate(denom_list):
            for r in term:
                ka_mat[i, self.rates.index(r)] = 1
        return ka_mat

    def gen_simulated_data(
        self,
        npoints: int = 1000,
        rng_seed: int = 1234,
        pheno_map: str = None,
        mut_num: Union[int, List[int], None] = None,
        save_hdf5: bool = False,
        **kwargs,
    ):
        """Generate simulated CSV with rates, activity and phenotype columns."""
        assert self.contrib_rate_names
        rng = np.random.default_rng(rng_seed)
        seq_arr = self.get_mutated_seqs(npoints, mut_num, rng)
        seq_values = self.model_params["Input"]["values"]
        if len(seq_values) == 2:
            seq_arr[1, :] = seq_values[1]

        rates_arr = np.zeros((npoints, len(self.rates)))
        act_arr = np.zeros(npoints)
        pheno_arr = np.zeros(npoints)
        pheno_map_func = eval(pheno_map) if pheno_map else (lambda x: x)

        for i, seq in enumerate(seq_arr):
            seq_ohe = self.generate_ohe_from_seq("".join(seq))
            for j, rate in enumerate(self.rates):
                rates_arr[i, j] = rate.get_rate(seq_ohe)
            for rate, occ in zip(self.contrib_rates, self.contrib_occupancy_funcs):
                act_arr[i] += rate.get_rate(seq_ohe) * occ(seq_ohe)
            pheno_arr[i] = pheno_map_func(act_arr[i])

        seq_list = ["".join(seq) for seq in seq_arr.tolist()]

        header = (
            ["seq"]
            + [r.name for r in self.rates]
            + ["raw_activity", "phenotype_activity"]
        )
        dtypes = ["S50"] + ["f4"] * (len(self.rates) + 2)
        dt = np.dtype([(name, dtype) for name, dtype in zip(header, dtypes)])

        data = np.zeros(npoints, dtype=dt)
        data["seq"] = seq_list
        for j, rate in enumerate(self.rates):
            data[rate.name] = rates_arr[:, j]
        data["raw_activity"] = act_arr
        data["phenotype_activity"] = pheno_arr

        if save_hdf5:
            self.save_hdf5_files(data)
            return

        self.save_csv_files(data)

    def save_csv_files(self, data) -> None:
        """Save CSV file with rates, activity and phenotype columns."""
        assert self.contrib_rate_names
        df = pd.DataFrame(data)
        df.columns = data.dtype.names
        for col in df.columns:
            if col != "seq":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_csv(
            Path(self.save_str + ".csv"), sep=",", float_format="%.5g", index=False
        )
        np.savetxt(
            Path(self.save_str + "_ka_mat.nptxt"), self.get_ka_pattern_mat(), fmt="%d"
        )
        np.savetxt(
            Path(self.save_str + "_rate_contrib_mat.nptxt"),
            self.get_rate_contrib_matrix(),
            fmt="%d",
        )

    def save_hdf5_files(self, data) -> None:
        """Save HDF5 file with rates, activity and phenotype columns."""
        # Create structured array with mixed types given data
        with h5py.File(Path(self.save_str + ".h5"), "w") as f:
            f.create_dataset("data", data=data)
            ka_mat = self.get_ka_pattern_mat()
            f.create_dataset("ka_pattern_matrix", data=ka_mat)
            contrib_mat = self.get_rate_contrib_matrix()
            f.create_dataset("rate_contrib_matrix", data=contrib_mat)
            f.attrs["params"] = yaml.dump(self.model_params)

    def get_rate_contrib_matrix(self) -> np.ndarray:
        denom_list = self.get_denominator()
        contrib_mat = np.zeros((len(self.rates), len(denom_list)), dtype=int)
        for i, crate in enumerate(self.rates):
            if crate.name in self.contrib_rate_names:
                start_state = crate.state_list[0]
                nterms = self.get_numerator(start_state)
                for j, kap in enumerate(denom_list):
                    if kap in nterms:
                        contrib_mat[i, j] = 1

        return contrib_mat
