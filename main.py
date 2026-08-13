import numpy as np
from system import CartPole
from utils import SinCosEncoder
from models import HierarchicalMPC, CosMPC


if __name__ == '__main__':
    # Cost matrices
    Q = np.diag([10.0, 1.0, 100.0, 1.0])
    R = np.diag([1.0])

    # Noise and sampling parameters
    sigma_X_simulate = np.array([0.005, 0.02, 0.005, 0.02])
    sigma_X_sample = sigma_X_simulate * 10
    sigma_U_sample = 1.0

    n_samples = 20_000


    # Initialize the cart-pole system
    system = CartPole(seed=1)

    system.generate_process_noise(sigma_X=np.diag(sigma_X_simulate) * 0)

    # Optional real system nmpc 
    # system.init_physical_mpc(Q=Q, R=R, Qt=Q*100, N=200)
    # pyhsical_mpc = system.physical_mpc
    # system.evaluate(control=pyhsical_mpc, Q=Q, R=R, Qt=Q*100)

    # Generate the optimal reference trajectory
    system.generate_reference(
        Q=Q,
        R=R, 
        x_max=np.array([1.0, np.inf, np.inf, np.inf]), 
        u_max=20
    )

    # Generate training data for the global model
    X_train, Xn_train, U_train = system.sample_reference(
        sigma_X=np.diag(sigma_X_sample),
        sigma_U=sigma_U_sample,
        n_samples=n_samples,
        H=10
    )

    # Train the global predictive model
    nmpc = SinCosEncoder(CosMPC(D=64, gamma=0.5, lam=0.5, layers=2, lr=1.0, wd=0.0, epochs=500))
    nmpc.fit(X_train, Xn_train, U_train.reshape(-1, 1))

    # Generate training data for the local model
    X_train, Xn_train, U_train = system.sample_reference(
        sigma_X=np.diag(sigma_X_sample),
        sigma_U=sigma_U_sample,
        n_samples=n_samples,
        H=1
    )

    # Train the local predictive model
    mpc = SinCosEncoder(CosMPC(D=64, gamma=0.5, lam=0.5, layers=2, lr=1.0, wd=0.0, epochs=500))
    mpc.fit(X_train, Xn_train, U_train.reshape(-1, 1))

    # Configure the hierarchical MPC controller
    control = HierarchicalMPC(
        nmpc, mpc, 
        Q_global=Q, Q_local=Q*10, 
        R_global=R, R_local=R*0.01, 
        Q_global_terminal=Q*100, 
        Rd_local=R*1.0, 
        x_terminal_ref=np.array([0, 0, 0, 0])
    )

    # Evaluate closed-loop performance
    system.evaluate(control=control, Q=Q, R=R, Qt=Q*100)