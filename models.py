import numpy as np
import torch
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import scipy



# Coordinate global planning with local MPC tracking.
class HierarchicalMPC:
    def __init__(self, global_model, local_model, Q_global, Q_local, R_global, R_local, Q_global_terminal, Rd_local, x_terminal_ref, N_global=10, N_local=10, M_global=10):
        self.global_model = global_model
        self.local_model = local_model
        self.Q_global = Q_global
        self.Q_local = Q_local
        self.R_global = R_global
        self.R_local = R_local
        self.Rd_local = Rd_local
        self.x_terminal_ref = x_terminal_ref
        self.N_global = N_global
        self.N_local = N_local
        self.M_global = M_global

        self.N = 0
        self.global_ref = None

        self.global_model.init_global_control(Q=Q_global, R=R_global, Qt=Q_global_terminal, N=N_global)
        self.local_model.init_local_control(Q=Q_local, R=R_local, Rd=Rd_local, N=N_local)



    # Compute the next hierarchical control input.
    def __call__(self, x):
        if self.global_ref is None or self.N >= self.M_global:
            u_global_ref, x_global_ref = self.global_model.global_control(x, return_state=True)

            self.global_ref = self.expand_global(x_global_ref, u_global_ref)

            self.N = 0

        u_global_ref_rep, x_global_ref_rep, w_global_ref_rep = self.global_ref

        X_ref = x_global_ref_rep[self.N : self.N + self.N_local]
        U_ref = u_global_ref_rep[self.N : self.N + self.N_local]
        W_ref = w_global_ref_rep[self.N : self.N + self.N_local]

        u = self.local_model.local_control(x, X_ref, U_ref, W_ref)

        self.N += 1

        return u

    # Expand global references with terminal-error correction.
    def expand_global_adj(self, X_global, U_global):
        if U_global.ndim == 1:
            U_global = U_global[:, None]

        n_segments = U_global.shape[0]

        X_start = X_global[: n_segments]
        X_target = X_global[1: n_segments + 1]
        U_segments = np.repeat(U_global[: n_segments, None, :], self.M_global, axis=1)

        X_forward = self.local_model.rollout_forward(X_start, U_segments)

        terminal_error = X_target - X_forward[:, -1]

        alpha = np.linspace(0.0, 1.0, self.M_global + 1)[None, :, None]

        X_segments = X_forward + alpha * terminal_error[:, None, :]

        X_segments[:, 0] = X_start
        X_segments[:, -1] = X_target

        X_fine = X_segments[:, :-1].reshape(-1, X_global.shape[1])
        U_fine = U_segments.reshape(-1, U_global.shape[1])
        
        W_fine = np.ones(n_segments * self.M_global)

        return U_fine, X_fine, W_fine

    # Expand global references to the local time scale.
    def expand_global(self, X_global, U_global):
        if U_global.ndim == 1:
            U_global = U_global[:, None]

        M = self.M_global
        n = len(U_global)

        U_segments = np.repeat(U_global[:, None, :], M, axis=1)
        X_segments = self.local_model.rollout_forward(X_global[:-1], U_segments)

        return (
            U_segments.reshape(n * M, -1),
            X_segments[:, :-1].reshape(n * M, -1),
            np.ones(n * M),
        )




# Shared acados MPC formulation and control routines.
class BaseMPC:
    def __init__(self, u_min=-20, u_max=20, x_terminal=np.array([0, 0, 0, 1, 0])):
        self.u_min = u_min
        self.u_max = u_max
        self.x_terminal = x_terminal
    
    # Initialize the local tracking MPC solver.
    def init_local_control(self, Q, R, Rd, N):
        p, nz = self.p, self.p + self.D
        na = nz + 1

        za, u = ca.MX.sym("za", na), ca.MX.sym("u", 1)
        z = za[:nz]
        v = (u - self.mu) / self.su

        model = AcadosModel()
        model.name = f"local_control"
        model.x, model.u = za, u
        model.disc_dyn_expr = ca.vertcat(ca.DM(self.W).T @ ca.vertcat(z, z * v, v), u)

        ocp = AcadosOcp()
        ocp.model = model

        ocp.cost.cost_type = ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.Vx = np.zeros((p + 2, na))
        ocp.cost.Vx[:p, :p] = np.eye(p)
        ocp.cost.Vx[p + 1:, nz:] = -1

        ocp.cost.Vu = np.zeros((p + 2, 1))
        ocp.cost.Vu[p] = 1
        ocp.cost.Vu[p + 1] = 1

        ocp.cost.W = scipy.linalg.block_diag(np.eye(p), R, Rd)
        ocp.cost.yref = np.zeros(p + 2)
        ocp.cost.Vx_e = np.zeros((0, na))
        ocp.cost.W_e = np.zeros((0, 0))
        ocp.cost.yref_e = np.zeros(0)

        ocp.constraints.x0 = np.zeros(na)
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.lbu = np.array([self.u_min])
        ocp.constraints.ubu = np.array([self.u_max])

        opt = ocp.solver_options
        opt.N_horizon = N
        opt.tf = N
        opt.integrator_type = "DISCRETE"
        opt.nlp_solver_type = "SQP_RTI"
        opt.qp_solver = "FULL_CONDENSING_HPIPM"
        opt.hessian_approx = "GAUSS_NEWTON"
        opt.print_level = 0

        self.solver = AcadosOcpSolver(ocp, json_file=f"local_control.json", verbose=False)

        S = np.diag(self.sx)
        self.Qs = S @ Q @ S
        self.R, self.Rd = R, Rd
        self.N = N

    # Initialize the global MPC solver.
    def init_global_control(self, Q, R, Qt, N):
        p, n = self.p, self.p + self.D

        z, u = ca.MX.sym("z", n), ca.MX.sym("u", 1)
        v = (u - self.mu) / self.su

        model = AcadosModel()
        model.name = f"global_control"
        model.x, model.u = z, u
        model.disc_dyn_expr = ca.DM(self.W).T @ ca.vertcat(z, z * v, v)

        xr = (self.x_terminal - self.mx) / self.sx
        S = np.diag(self.sx)

        ocp = AcadosOcp()
        ocp.model = model

        ocp.cost.cost_type = ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.Vx = np.r_[np.eye(p, n), np.zeros((1, n))]
        ocp.cost.Vu = np.r_[np.zeros((p, 1)), [[1]]]
        ocp.cost.W = scipy.linalg.block_diag(S @ Q @ S, R)
        ocp.cost.yref = np.r_[xr, 0]
        ocp.cost.Vx_e = np.eye(p, n)
        ocp.cost.W_e = S @ Qt @ S
        ocp.cost.yref_e = xr

        ocp.constraints.x0 = np.zeros(n)
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.lbu = np.array([self.u_min])
        ocp.constraints.ubu = np.array([self.u_max])

        opt = ocp.solver_options
        opt.N_horizon = N
        opt.tf = N
        opt.integrator_type = "DISCRETE"
        opt.nlp_solver_type = "SQP_RTI"
        opt.qp_solver = "FULL_CONDENSING_HPIPM"
        opt.hessian_approx = "GAUSS_NEWTON"
        opt.print_level = 0

        self.solver = AcadosOcpSolver(ocp, json_file=f"global_control.json", verbose=False)

        self.N = N

    # Solve the local reference-tracking problem.
    def local_control(self, x, X_ref, U_ref, W_ref, return_state=False):
        Qs, R, Rd = self.Qs, self.R, self.Rd
        p, nz = self.p, self.p + self.D
        na, N = nz + 1, self.N

        s = self.solver

        z0 = self.lift_np(((x - self.mx) / self.sx)[None]).reshape(nz)
        za0 = np.r_[z0, getattr(self, "u_prev", U_ref[0])]

        s.set(0, "lbx", za0)
        s.set(0, "ubx", za0)

        X_ref_n = (X_ref - self.mx) / self.sx

        for k in range(N):
            s.cost_set(k, "W", scipy.linalg.block_diag(W_ref[k] * Qs, R, Rd))
            s.set(k, "yref", np.r_[X_ref_n[k], U_ref[k], 0])

        ZA = np.zeros((N + 1, na))
        ZA[0] = za0

        for k in range(N):
            z = ZA[k, :nz]
            v = (U_ref[k, 0] - self.mu) / self.su
            ZA[k + 1] = np.r_[np.r_[z, z * v, v] @ self.W, U_ref[k, 0]]


        for k in range(N):
            s.set(k, "x", ZA[k])
            s.set(k, "u", U_ref[k])

        s.set(N, "x", ZA[-1])

        s.solve()

        self.U = np.array([s.get(k, "u") for k in range(N)])
        self.ZA = np.array([s.get(k, "x") for k in range(N + 1)])
        self.u_prev = self.U[0]

        X = self.ZA[:, :p] * self.sx + self.mx

        return (self.U, X) if return_state else self.U[0]
    
    # Solve the global control problem.
    def global_control(self, x, return_state=False):
        p, n, N = self.p, self.p + self.D, self.N
  
        s = self.solver
        z0 = self.lift_np(((x - self.mx) / self.sx)[None]).reshape(n)

        U = getattr(self, "U", np.zeros((N, 1)))
        U = np.clip(np.r_[U[1:], U[-1:]], self.u_min, self.u_max)

        Z = getattr(self, "Z", None)

        if Z is not None:
            Z = np.r_[Z[1:], Z[-1:]]
            Z[0] = z0
        else:
            Z = np.zeros((N + 1, n))
            Z[0] = z0
            for k in range(N):
                v = (U[k, 0] - self.mu) / self.su
                Z[k + 1] = np.r_[Z[k], Z[k] * v, v] @ self.W

        s.set(0, "lbx", z0)
        s.set(0, "ubx", z0)

        for k in range(N):
            s.set(k, "x", Z[k])
            s.set(k, "u", U[k])

        s.set(N, "x", Z[N])
        s.solve()

        self.U = np.array([s.get(k, "u") for k in range(N)])
        self.Z = np.array([s.get(k, "x") for k in range(N + 1)])

        X = self.Z[:, :p] * self.sx + self.mx

        return (self.U, X) if return_state else self.U[0]







# Shared rollout utilities for the lifted dynamics.
class BaseZ:
    def __init__(self):
        pass

    # Propagate lifted dynamics forward.
    def rollout_forward(self, x0, U):
        if U.ndim == 2:
            U = U[None]

        S, N, _ = U.shape

        if x0.ndim == 1:
            x0 = np.repeat(x0[None], S, axis=0)

        X = np.empty((S, N + 1, self.p))
        X[:, 0] = x0

        z = self.lift_np((x0 - self.mx) / self.sx)

        for k in range(N):
            u = U[:, k]
            un = (u - self.mu) / self.su

            z = np.c_[z, z * un, un] @ self.W
            X[:, k + 1] = z[:, :self.p] * self.sx + self.mx

        return X


    # Reconstruct a trajectory backward with regularization.
    def rollout_backward(self, xN, U, regularization=1e-6):
        if U.ndim == 2:
            U = U[None]

        S, N, _ = U.shape

        if xN.ndim == 1:
            xN = np.repeat(xN[None], S, axis=0)

        nz = self.p + self.D

        Wz = self.W[:nz]
        Wzu = self.W[nz:2 * nz]
        Wu = self.W[2 * nz:]

        X = np.empty((S, N + 1, self.p))
        X[:, N] = xN

        z = self.lift_np((xN - self.mx) / self.sx)

        for k in range(N - 1, -1, -1):
            un = ((U[:, k] - self.mu) / self.su).reshape(S)

            z_previous = np.empty_like(z)

            for s in range(S):
                A = Wz + un[s] * Wzu
                b = (un[s] * Wu).reshape(nz)
                rhs = z[s] - b

                M = A @ A.T + regularization * np.eye(nz)

                z_previous[s] = np.linalg.solve(
                    M,
                    A @ rhs,
                )

            z = z_previous
            X[:, k] = z[:, :self.p] * self.sx + self.mx

        return X








# Cosine-feature lifted model for data-driven MPC.
class CosMPC(BaseMPC, BaseZ):
    def __init__(self, D, gamma, lam, layers=1, lr=1e-3, wd=1e-2,
                 epochs=0, seed=42):
        super().__init__()
        self.D, self.gamma, self.lam = D, gamma, lam
        self.layers, self.lr, self.wd, self.epochs = layers, lr, wd, epochs
        self.rng = np.random.default_rng(seed)
        self.seed, self.eps = seed, 1e-8

    # Fit the lifted linear dynamics model.
    def fit(self, X, Xn, U):
        self.p = X.shape[1]
        self.mx, self.sx = X.mean(0), np.sqrt(X.var(0) + self.eps)
        self.mu, self.su = U.mean(0), np.sqrt(U.var(0) + self.eps)

        X, Xn = (X - self.mx) / self.sx, (Xn - self.mx) / self.sx
        U = (U - self.mu) / self.su

        dims = [self.p] + [self.D] * (self.layers - 1)
        self.omegas = [
            self.rng.normal(0, np.sqrt(2 * self.gamma), (d, self.D))
            for d in dims
        ]
        self.betas = [
            self.rng.uniform(0, 2 * np.pi, self.D)
            for _ in dims
        ]

        if self.epochs:
            torch.manual_seed(self.seed)
            X, Xn, U = map(
                lambda a: torch.tensor(a, dtype=torch.float32),
                (X, Xn, U),
            )
            ws = torch.nn.ParameterList([
                torch.nn.Parameter(torch.tensor(w, dtype=torch.float32))
                for w in self.omegas
            ])
            bs = torch.nn.ParameterList([
                torch.nn.Parameter(torch.tensor(b, dtype=torch.float32))
                for b in self.betas
            ])
            opt = torch.optim.AdamW(
                [*ws.parameters(), *bs.parameters()],
                lr=self.lr,
                weight_decay=self.wd,
            )

            for epoch in range(self.epochs):
                opt.zero_grad()
                Z, Zn = self._lift(X, ws, bs), self._lift(Xn, ws, bs)
                H = torch.cat((Z, Z * U, U), 1)
                W = torch.linalg.solve(
                    H.T @ H + self.lam * torch.eye(H.shape[1]),
                    H.T @ Zn,
                )
                loss = ((H @ W - Zn) ** 2).mean()
                loss.backward()
                opt.step()

                if epoch % 10 == 0 or epoch == self.epochs - 1:
                    print(f"epoch {epoch} | loss {loss.item():.6e}")

            self.omegas = [w.detach().numpy() for w in ws]
            self.betas = [b.detach().numpy() for b in bs]
            X, Xn, U = map(lambda a: a.numpy(), (X, Xn, U))

        Z, Zn = self.lift_np(X), self.lift_np(Xn)
        H = np.c_[Z, Z * U, U]
        self.W = np.linalg.solve(
            H.T @ H + self.lam * np.eye(H.shape[1]),
            H.T @ Zn,
        )

        print(f"loss {np.mean((H @ self.W - Zn) ** 2):.6e}")
        return self

    # Apply the trainable cosine lifting map.
    def _lift(self, X, omegas, betas):
        h = X
        c = (2 / self.D) ** 0.5

        for w, b in zip(omegas, betas):
            h = torch.cos(h @ w + b) * c

        return torch.cat((X, h), dim=1)

    # Apply the cosine lifting map with NumPy.
    def lift_np(self, X):
        X = np.atleast_2d(X)
        h = X
        c = (2 / self.D) ** 0.5

        for w, b in zip(self.omegas, self.betas):
            h = np.cos(h @ w + b) * c

        return np.concatenate((X, h), axis=1)
    
    # Measure correlation among lifted coordinates.
    def latent_redundancy(self, X):
        Z = self.lift_np((X - self.mx) / self.sx)
        C = np.corrcoef(Z, rowvar=False)
        np.fill_diagonal(C, 0)
        return np.max(np.abs(C), axis=1)

