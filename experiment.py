import numpy as np
from system import CartPole
from utils import SinCosEncoder
from models import HierarchicalMPC, CosMPC

import os
import shutil
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config


def run_experiment(seed):
    # Use an isolated working directory for each acados process.
    workdir = Path(".acados_jobs") / f"seed_{seed}_pid_{os.getpid()}"
    workdir.mkdir(parents=True, exist_ok=True)

    original_directory = Path.cwd()
    os.chdir(workdir)

    try:
        # Cost and sampling parameters
        Q = np.diag([10.0, 1.0, 100.0, 1.0])
        R = np.diag([1.0])

        sigma_X_simulate = np.array([0.005, 0.02, 0.005, 0.02])
        sigma_X_sample = sigma_X_simulate * 10
        sigma_U_sample = 1.0
        n_samples = 20_000

        # Generate independent seeds for each model component.
        child_seeds = np.random.SeedSequence(seed).spawn(3)
        system_seed, nmpc_seed, mpc_seed = [
            int(s.generate_state(1)[0])
            for s in child_seeds
        ]

        # Initialize the cart-pole system.
        system = CartPole(seed=system_seed)

        system.generate_process_noise(
            sigma_X=np.diag(sigma_X_simulate) * 0
        )

        # Evaluate the physical-model MPC baseline.
        system.init_physical_mpc(
            Q=Q,
            R=R,
            Qt=Q * 100,
            N=200,
        )

        physical_metrics = system.evaluate(
            control=system.physical_mpc,
            Q=Q,
            R=R,
            Qt=Q * 100,
            animate=False,
        )

        # Generate the reference trajectory.
        system.generate_reference(
            Q=Q,
            R=R,
            x_max=np.array([1.0, np.inf, np.inf, np.inf]),
            u_max=20,
        )

        # Generate training data for the global model.
        X, Xn, U = system.sample_reference(
            sigma_X=np.diag(sigma_X_sample),
            sigma_U=sigma_U_sample,
            n_samples=n_samples,
            H=10,
        )

        # Train the global predictive model.
        nmpc = SinCosEncoder(
            CosMPC(
                D=128,
                gamma=0.5,
                lam=0.5,
                layers=1,
                lr=1.0,
                wd=0.0,
                epochs=500,
                seed=nmpc_seed,
            )
        )
        nmpc.fit(X, Xn, U.reshape(-1, 1))

        # Generate training data for the local model.
        X, Xn, U = system.sample_reference(
            sigma_X=np.diag(sigma_X_sample),
            sigma_U=sigma_U_sample,
            n_samples=n_samples,
            H=1,
        )

        # Train the local predictive model.
        mpc = SinCosEncoder(
            CosMPC(
                D=128,
                gamma=0.5,
                lam=0.5,
                layers=1,
                lr=1.0,
                wd=0.0,
                epochs=500,
                seed=mpc_seed,
            )
        )
        mpc.fit(X, Xn, U.reshape(-1, 1))

        # Configure and evaluate the hierarchical MPC.
        control = HierarchicalMPC(
            nmpc,
            mpc,
            Q_global=Q,
            Q_local=Q * 10,
            R_global=R,
            R_local=R * 0.01,
            Q_global_terminal=Q * 100,
            Rd_local=R,
            x_terminal_ref=np.zeros(4),
        )

        hierarchical_metrics = system.evaluate(
            control=control,
            Q=Q,
            R=R,
            Qt=Q * 100,
            animate=False,
        )

        return {
            "seed": seed,
            "physical": physical_metrics,
            "hierarchical": hierarchical_metrics,
        }

    finally:
        # Restore the working directory and remove temporary files.
        os.chdir(original_directory)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    os.environ["ACADOS_SOURCE_DIR"] = "/home/lennart/acados"

    seeds = range(30)

    # Run independent experiments in parallel.
    with parallel_config(
        backend="loky",
        inner_max_num_threads=1,
    ):
        results = Parallel(
            n_jobs=10,
            verbose=10,
        )(
            delayed(run_experiment)(seed)
            for seed in seeds
        )

    # Report aggregate metrics across all seeds.
    for controller, metrics in results[0].items():
        if not isinstance(metrics, dict):
            continue

        print(f"\n=== {controller.upper()} ===")

        for metric in metrics:
            values = np.asarray([
                result[controller][metric]
                for result in results
            ])

            print(
                f"{metric:20s}: "
                f"{values.mean():.6f} ± {values.std():.6f}"
            )