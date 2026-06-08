import numpy as np
from system import CartPole
from utils import SinCosEncoder
from models import RFFMPC, LearnedRFFMPC, DeepRFFMPC


if __name__ == '__main__':
    Q = np.diag([50.0, 1.0, 100.0, 1.0])
    R = np.diag([1.0])
    Qe = SinCosEncoder.convert_Q(Q)

    sigma_X_simulate = np.array([0.005, 0.02, 0.005, 0.02])
    sigma_X_sample = sigma_X_simulate * 100

    H = 50


    system = CartPole()

    system.generate_reference(
        Q=Q,
        R=R, 
        x_max=np.array([1.0, np.inf, np.inf, np.inf]), 
        u_max=20
    )

    X_train, Xn_train, U_train = system.sample_reference(
        sigma_X=np.diag(sigma_X_sample),
        sigma_U=1.0,
        n_samples_per_step=32,
    )

    system.generate_process_noise(sigma_X=np.diag(sigma_X_simulate))




    # RFFMPC
    print('Training RFF.')

    mpc = SinCosEncoder(RFFMPC(
        D=256,
        gamma=0.5,
        lam=1.0,
        Q=Qe,
        R=R
    ))

    mpc.fit(X_train, Xn_train, U_train.reshape(-1, 1))

    system.simulate(mpc.MPC, H=H)



    # LearnedRFFMPC
    print('Training Learned RFF.')

    mpc = SinCosEncoder(LearnedRFFMPC(
        D=256,
        gamma=0.5,
        lam=1.0,
        Q=Qe,
        R=R
    ))

    mpc.fit(X_train, Xn_train, U_train.reshape(-1, 1))

    system.simulate(mpc.MPC, H=H)



    # DeepRFFMPC
    print('Training Deep RFF.')

    mpc = SinCosEncoder(DeepRFFMPC(
        D=256,
        gamma=0.5,
        lam=1.0,
        Q=Qe,
        R=R
    ))

    mpc.fit(X_train, Xn_train, U_train.reshape(-1, 1))

    system.simulate(mpc.MPC, H=H)