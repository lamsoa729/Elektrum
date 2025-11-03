import sys
from pathlib import Path
import numpy as np
import h5py
import time
from scipy.optimize import minimize

from theory_helper_funcs import (
    activity_func_generator,
    min_mse_loss,
    min_mse_loss_reg,
    extend_sequences,
    load_kinetic_data,
)


def infer_parameters(x, y, ka_mat, ka_contrib, bounds=(-10, 4)):
    """
    Infer kinetic parameters using regression on King-Altman model.

    Args:
        x: Input data (one-hot encoded sequences)
        y: Observed activities
        ka_mat: King-Altman matrix
        ka_contrib: Contribution matrix for activity calculation
        regularization: Regularization strength

    Returns:
        Optimized parameters
    """
    f = activity_func_generator(ka_mat, ka_contrib)
    n = x.shape[1]

    # Initial guess for parameters
    # initial_theta = np.random.uniform(bounds[0], bounds[1], size=2 * n)
    initial_theta = [0] * (2 * n)

    # Minimize the regularized MSE loss
    result = minimize(
        min_mse_loss,
        initial_theta,
        args=(f, x, y),
        method="L-BFGS-B",
        bounds=[bounds] * (2 * n),
    )

    return result.x


if __name__ == "__main__":
    # Get files from a directory
    dataset_dir = Path(sys.argv[1])
    dataset_files = list(dataset_dir.glob("*.h5"))
    # sort dataset_files based on number at the end of the filename before .h5
    dataset_files = sorted(dataset_files, key=lambda x: int(x.stem.split("_")[-1]))

    rate_matrices = []
    for i in range(len(dataset_files)):
        # Time each iteration
        start_time = time.perf_counter()
        dataset_file = dataset_files[i]

        data = load_kinetic_data(dataset_file)
        x = data["x"]
        y = data["y"]
        # rate_names = data["rate_names"]
        # print("Rate names:", rate_names)

        optimized_params = infer_parameters(x, y, data["ka_mat"], data["ka_contrib"])

        rate_names = data["dtype"].names[1:-2]
        # rate_types = np.dtype([(name, "f4") for name in rate_names])

        inferred_targ_rates = np.exp(optimized_params[::2])
        inferred_mut_rates = np.exp(optimized_params[1::2])

        true_targ_rates = data["target_rates"]
        true_mut_rates = data["mut_rates"]

        # print("Inferred Target Rates:", inferred_targ_rates)
        # print("True Target Rates:", true_targ_rates)
        # print("Inferred Mutant Rates:", inferred_mut_rates)
        # print("True Mutant Rates:", true_mut_rates)
        rate_mat = np.vstack(
            [
                inferred_targ_rates,
                true_targ_rates,
                inferred_mut_rates,
                true_mut_rates,
            ]
        )
        rate_matrices.append(rate_mat)
        end_time = time.perf_counter()
        print(f"Dataset {i} processed in {end_time - start_time:.2f} seconds.")

    with h5py.File(
        dataset_dir.parent / f"{dataset_dir.name}_inferred_params.h5", "w"
    ) as f:
        stacked_rates = np.vstack([rate_matrices])
        print("Stacked rates shape:", stacked_rates.shape)
        stacked_rates = np.einsum("ijk->jki", stacked_rates)
        print("Stacked rates shape:", stacked_rates.shape)
        # Stack infered and true rates for comparison
        f.create_dataset(
            "rates",
            data=stacked_rates,
        )
