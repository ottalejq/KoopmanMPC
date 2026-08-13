import numpy as np
import casadi as ca
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
from scipy.linalg import expm, solve_discrete_are
import time
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import scipy



# Plot styling
mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,

    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,

    "figure.dpi": 150,
    "savefig.dpi": 300,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})



# Cart-pole dynamics, simulation, sampling, and MPC utilities
class CartPole:
    def __init__(
        self,
        T=10,
        dt=1/50,
        n_substeps=5,
        M=1.0,
        m=1.0,
        l=1.0,
        g=9.81,
        seed=None
    ):
        self.T = T
        self.dt = dt
        self.n_substeps = n_substeps

        self.Xf = np.zeros(4)
        self.X0 = np.array([0, 0, np.pi, 0])

        self.N = int(self.T / self.dt)

        self.M = M
        self.m = m
        self.l = l
        self.g = g

        self.rng = np.random.default_rng(seed)


    # Compute the discrete-time terminal LQR cost.
    def Pt(self, Q, R):
        A_c = np.array([
            [0, 1, 0, 0],
            [0, 0, -self.m * self.g / self.M, 0],
            [0, 0, 0, 1],
            [0, 0, (self.M + self.m) * self.g / (self.l * self.M), 0],
        ])

        B_c = np.array([
            [0],
            [1 / self.M],
            [0],
            [-1 / (self.l * self.M)],
        ])

        n = A_c.shape[0]
        m = B_c.shape[1]

        M = np.zeros((n + m, n + m))
        M[:n, :n] = A_c
        M[:n, n:] = B_c

        Md = expm(M * self.dt)

        A_d = Md[:n, :n]
        B_d = Md[:n, n:]

        Q_d = Q * self.dt
        R_d = R * self.dt

        P = solve_discrete_are(A_d, B_d, Q_d, R_d)

        return P

    # Evaluate continuous-time cart-pole dynamics.
    def dynamics(self, z, u, xp=np):
        if xp is np:
            z = np.asarray(z)
            u = np.asarray(u).reshape(-1)

            single = z.ndim == 1
            Z = z[None, :] if single else z

            x, xd, th, thd = Z.T
            u = u[0] if single else u

            s, c = np.sin(th), np.cos(th)
            d = self.M + self.m * s**2

            xdd = (u - self.m*self.g*s*c + self.m*self.l*s*thd**2) / d
            thdd = (-u*c + (self.M+self.m)*self.g*s - self.m*self.l*s*c*thd**2) / (self.l*d)

            out = np.c_[xd, xdd, thd, thdd]
            return out[0] if single else out

        x, xd, th, thd = z[0], z[1], z[2], z[3]
        u = u[0]

        s, c = xp.sin(th), xp.cos(th)
        d = self.M + self.m * s**2

        xdd = (u - self.m*self.g*s*c + self.m*self.l*s*thd**2) / d
        thdd = (-u*c + (self.M+self.m)*self.g*s - self.m*self.l*s*c*thd**2) / (self.l*d)

        return xp.vertcat(xd, xdd, thd, thdd)

    # Integrate one time step with RK4.
    def step(self, z, u):
        one = z.ndim == 1
        z = z[None].copy() if one else z.copy()

        h = self.dt / self.n_substeps

        for _ in range(self.n_substeps):
            k1 = self.dynamics(z, u)
            k2 = self.dynamics(z + 0.5 * h * k1, u)
            k3 = self.dynamics(z + 0.5 * h * k2, u)
            k4 = self.dynamics(z + h * k3, u)

            z += h * (k1 + 2*k2 + 2*k3 + k4) / 6

        return z[0] if one else z

    # CasADi-compatible RK4 integration.
    def step_casadi(self, z, u):
        h = self.dt / self.n_substeps
        zn = z

        for _ in range(self.n_substeps):
            k1 = self.dynamics(zn, u, xp=ca)
            k2 = self.dynamics(zn + 0.5 * h * k1, u, xp=ca)
            k3 = self.dynamics(zn + 0.5 * h * k2, u, xp=ca)
            k4 = self.dynamics(zn + h * k3, u, xp=ca)

            zn = zn + h * (k1 + 2*k2 + 2*k3 + k4) / 6

        return zn

    # Generate random state-control trajectories.
    def generate_random_sequences(self, K, H, x_low, x_high, u_low, u_high):
        x = self.rng.uniform(x_low, x_high, size=(K, 4))
        U = self.rng.uniform(u_low, u_high, size=(K, H, 1))

        X = np.empty((K, H, 4))
        Xn = np.empty((K, H, 4))

        for t in range(H):
            X[:, t] = x
            x = self.step(x, U[:, t])
            Xn[:, t] = x

        return X, Xn, U

    # Solve for a constrained reference trajectory and feedback gains.
    def generate_reference(self, Q, R, x_max, u_max):
        print('Generating reference trajectory.')

        opti = ca.Opti()

        nx, nu = 4, 1
        X = opti.variable(nx, self.N + 1)
        U = opti.variable(nu, self.N)

        opti.subject_to(X[:, 0] == self.X0)

        for k in range(self.N):
            opti.subject_to(X[:, k + 1] == self.step_casadi(X[:, k], U[:, k]))

        opti.subject_to(opti.bounded(-x_max, X, x_max))
        opti.subject_to(opti.bounded(-u_max, U, u_max))

        cost = 0
        for k in range(self.N):
            dX = X[:, k] - self.Xf
            cost += ca.mtimes([dX.T, Q, dX]) * self.dt
            cost += ca.mtimes([U[:, k].T, R, U[:, k]]) * self.dt

        P = self.Pt(Q, R)

        dXf = X[:, -1] - self.Xf
        cost += ca.mtimes([dXf.T, P, dXf])

        opti.minimize(cost)

        t_grid = np.linspace(0.0, self.T, self.N + 1)
        opti.set_initial(X[2, :], np.pi * (1 + np.cos(np.pi * t_grid / self.T)) / 2)
        opti.set_initial(U, 0.0)

        opti.solver(
            "ipopt",
            {
                "expand": True,
                "print_time": False,
            },
            {
                "print_level": 0,
                "sb": "yes",
                "print_user_options": "no",
                "print_options_documentation": "no",
                "tol": 1e-6,
                "max_iter": 3000,
                "acceptable_tol": 1e-5,
                "mu_strategy": "adaptive",
            },
        )

        sol = opti.solve()

        X_ref = sol.value(X).T
        U_ref = sol.value(U).reshape(self.N, nu)

        x_sym = ca.MX.sym("x", nx)
        u_sym = ca.MX.sym("u", nu)
        f_sym = self.step_casadi(x_sym, u_sym)

        A_fun = ca.Function("A_fun", [x_sym, u_sym], [ca.jacobian(f_sym, x_sym)])
        B_fun = ca.Function("B_fun", [x_sym, u_sym], [ca.jacobian(f_sym, u_sym)])

        K = np.zeros((self.N, nu, nx))

        A_seq = np.zeros((self.N, nx, nx))
        B_seq = np.zeros((self.N, nx, nu))
        K = np.zeros((self.N, nu, nx))

        for k in reversed(range(self.N)):
            x = X_ref[k, :]
            u = U_ref[k, :]

            A = np.asarray(A_fun(x, u), dtype=float)
            B = np.asarray(B_fun(x, u), dtype=float)

            A_seq[k] = A
            B_seq[k] = B

            S = R + B.T @ P @ B
            K[k, :, :] = np.linalg.solve(S, B.T @ P @ A)

            P = Q + A.T @ P @ (A - B @ K[k, :, :])

        self.X_ref = X_ref
        self.U_ref = U_ref
        self.K = K
        self.A_seq = A_seq
        self.B_seq = B_seq

        return self


    # Sample training data around the reference trajectory.
    def sample_reference(
        self,
        sigma_X,
        sigma_U,
        n_samples,
        H
    ):
        print('Generating training samples.')

        X, Xn, U = [], [], []

        n_samples_per_step = n_samples // self.N

        for x_ref, u_ref, K in zip(self.X_ref, self.U_ref, self.K):
            dx = self.rng.multivariate_normal(
                mean=np.zeros(x_ref.shape),
                cov=sigma_X**2,
                size=n_samples_per_step,
            )

            du = self.rng.normal(
                loc=0.0,
                scale=sigma_U,
                size=n_samples_per_step,
            )

            x = x_ref + dx
            u = u_ref - dx @ K.reshape(-1) + du

            xn = x.copy()
            for _ in range(H):
                xn = self.step(xn, u)

            X.append(x)
            Xn.append(xn)
            U.append(u)

        return (
            np.vstack(X),
            np.vstack(Xn),
            np.concatenate(U),
        )


    # Sample random states and constant-input rollouts.
    def sample_random(
        self,
        n_samples,
        H,
        x_low=None,
        x_high=None,
        u_low=-20.0,
        u_high=20.0,
    ):

        print("Generating random training samples.")

        if x_low is None:
            x_low = np.array([-2.0, -5.0, -np.pi, -5.0])

        if x_high is None:
            x_high = np.array([2.0, 5.0, np.pi, 5.0])

        X = self.rng.uniform(
            low=x_low,
            high=x_high,
            size=(n_samples, len(x_low)),
        )

        U = self.rng.uniform(
            low=u_low,
            high=u_high,
            size=(n_samples, 1),
        )

        Xn = X.copy()
        for _ in range(H):
            Xn = self.step(Xn, U)

        return X, Xn, U

    # Pre-generate process noise for simulation.
    def generate_process_noise(self, sigma_X):
        self.process_noise = self.rng.multivariate_normal(
            mean=np.zeros(4),
            cov=sigma_X**2,
            size=self.N
        )
        return self

    # Simulate the closed-loop system.
    def simulate(self, control, animate=True):
        x = self.X0

        X = np.zeros((self.N, 4))
        Xn = np.zeros((self.N, 4))
        U = np.zeros((self.N, 1))

        control_time = 0.0

        for k in range(self.N):   
            t0 = time.perf_counter()

            u = control(x)

            control_time += time.perf_counter() - t0

            u = float(np.asarray(u).squeeze())

            xn = self.step(x, u)

            xn += self.process_noise[k]

            X[k] = x
            Xn[k] = xn
            U[k, 0] = u

            x = xn

        print()
        print(f"Mean MPC control time: {1e3 * control_time / self.N:.3f} ms")

        if animate:
            self.ani = self.animate(X, U)

        return X, Xn, U

    # Evaluate closed-loop performance metrics.
    def evaluate(self, control, Q, R, Qt, animate=True, tol=0.05, angle_tol=np.deg2rad(10), rate_tol=0.5):
        X, Xn, U = self.simulate(control, animate)

        E = X - self.Xf
        state_cost = np.sum(np.einsum("ni,ij,nj->n", E, Q, E))
        input_cost = np.sum(np.einsum("ni,ij,nj->n", U, R, U))

        terminal_error = Xn[-1] - self.Xf
        terminal_cost = terminal_error @ Qt @ terminal_error
        total_cost = state_cost + input_cost + terminal_cost

        rmse = np.sqrt(np.mean(E**2, axis=0))
        error_norm = np.linalg.norm(Xn - self.Xf, axis=1)

        suffix = np.minimum.accumulate((error_norm < tol)[::-1])[::-1]

        settling_time = (
            self.dt * np.argmax(suffix)
            if np.any(suffix)
            else np.nan
        )

        theta_error = np.arctan2(
            np.sin(Xn[-1, 2] - self.Xf[2]),
            np.cos(Xn[-1, 2] - self.Xf[2]),
        )

        success = (
            abs(theta_error) < angle_tol
            and abs(Xn[-1, 3] - self.Xf[3]) < rate_tol
        )

        metrics = {
            "success": float(success),
            "total_cost": float(total_cost),
            "state_cost": float(state_cost),
            "input_cost": float(input_cost),
            "terminal_cost": float(terminal_cost),
            "final_error": float(np.linalg.norm(terminal_error)),
            "rmse_x": float(rmse[0]),
            "rmse_xdot": float(rmse[1]),
            "rmse_theta": float(rmse[2]),
            "rmse_thetadot": float(rmse[3]),
            "control_effort": float(np.sum(U[:, 0] ** 2)),
            "control_variation": float(np.sum(np.diff(U[:, 0]) ** 2)),
            "max_control": float(np.max(np.abs(U[:, 0]))),
            "settling_time": float(settling_time),
        }

        for metric, value in metrics.items():
            print(f'{metric}: {value}')

        return metrics

    # Initialize the acados nonlinear MPC solver.
    def init_physical_mpc(self, Q, R, Qt, N, u_min=-20.0, u_max=20.0, x_ref=None):
        nx, nu = 4, 1
        Q = np.asarray(Q, dtype=float)
        R = np.atleast_2d(np.asarray(R, dtype=float))
        Qt = np.asarray(Qt, dtype=float)
        x_ref = self.Xf if x_ref is None else np.asarray(x_ref, dtype=float).reshape(nx)

        x, u = ca.MX.sym("x", nx), ca.MX.sym("u", nu)

        model = AcadosModel()
        model.name = "true_control"
        model.x, model.u = x, u
        model.disc_dyn_expr = self.step_casadi(x, u)

        ocp = AcadosOcp()
        ocp.model = model

        ocp.cost.cost_type = ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.Vx = np.r_[np.eye(nx), np.zeros((nu, nx))]
        ocp.cost.Vu = np.r_[np.zeros((nx, nu)), np.eye(nu)]
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.yref = np.r_[x_ref, np.zeros(nu)]
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.W_e = Qt
        ocp.cost.yref_e = x_ref

        ocp.constraints.x0 = np.zeros(nx)
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.lbu = np.array([u_min])
        ocp.constraints.ubu = np.array([u_max])

        opt = ocp.solver_options
        opt.N_horizon = N
        opt.tf = N * self.dt
        opt.integrator_type = "DISCRETE"
        opt.nlp_solver_type = "SQP_RTI"
        opt.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        opt.hessian_approx = "GAUSS_NEWTON"
        opt.print_level = 0

        self.physical_mpc_solver = AcadosOcpSolver(ocp, json_file="true_control.json", verbose=False)
        self.physical_mpc_N = N
        self.physical_mpc_x_ref = x_ref
        self.physical_mpc_u_min = u_min
        self.physical_mpc_u_max = u_max
        self.physical_mpc_U = np.zeros((N, nu))
        self.physical_mpc_X = None


    # Solve one MPC step using warm-started trajectories.
    def physical_mpc(self, x, return_state=False):
        nx, N = 4, self.physical_mpc_N
        s = self.physical_mpc_solver

        x = np.asarray(x, dtype=float)

        s.set(0, "lbx", x)
        s.set(0, "ubx", x)

        U = np.clip(
            np.r_[self.physical_mpc_U[1:], self.physical_mpc_U[-1:]],
            self.physical_mpc_u_min,
            self.physical_mpc_u_max,
        )

        if self.physical_mpc_X is None:
            X = np.zeros((N + 1, nx))
            X[0] = x
            for k in range(N):
                X[k + 1] = self.step(X[k], U[k])
        else:
            X = np.r_[self.physical_mpc_X[1:], self.physical_mpc_X[-1:]]
            X[0] = x

        for k in range(N):
            s.set(k, "x", X[k])
            s.set(k, "u", U[k])

        s.set(N, "x", X[N])

        s.solve()

        self.physical_mpc_U = np.array([s.get(k, "u") for k in range(N)])
        self.physical_mpc_X = np.array([s.get(k, "x") for k in range(N + 1)])

        return (self.physical_mpc_U, self.physical_mpc_X) if return_state else self.physical_mpc_U[0]



    # Animate the cart-pole trajectory and applied force.
    def animate(self, X=None, U=None):
        cw, ch, wr = 0.4, 0.2, 0.05
        view = 2.5
        force_scale = 0.05

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set(aspect="equal", ylim=(-0.3, 1.5))
        ax.axhline(0, color="black")

        cart = ax.add_patch(
            FancyBboxPatch((-cw / 2, 2 * wr), cw, ch, boxstyle="round,pad=0.02", facecolor="white", edgecolor="black", lw=2, zorder=0)
        )
        w1 = ax.add_patch(plt.Circle((-cw / 4, wr), wr, facecolor="white", edgecolor="black", lw=2, zorder=1))
        w2 = ax.add_patch(plt.Circle((cw / 4, wr), wr, facecolor="white", edgecolor="black", lw=2, zorder=1))

        rod, = ax.plot([], [], color="black", lw=2, zorder=2)
        mass = ax.add_patch(plt.Circle((0, 0), 0.08, facecolor="white", edgecolor="black", lw=2, zorder=4))
        txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, color="black")

        force_ax = ax.inset_axes([0.62, 0.72, 0.34, 0.22])
        force_ax.set_title("u(t)", fontsize=9, color="black")
        force_ax.axhline(0, color="black", lw=1)

        force_line, = force_ax.plot([], [], color="black")
        force_dot, = force_ax.plot([], [], "o", color="black")
        force_arrow = [None]

        def draw(z, u, k):
            x, th = z[0], z[2]
            t = k * self.dt

            px, py = x, wr + ch
            mx = px + self.l * np.sin(th)
            my = py + self.l * np.cos(th)

            ax.set_xlim(x - view, x + view)

            pivot_y = 2 * wr + ch
            cart.set_bounds(x - cw / 2, 2 * wr, cw, ch)
            w1.center = (x - 0.3 * cw, wr)
            w2.center = (x + 0.3 * cw, wr)
            px = x
            py = pivot_y + 0.025

            mx = px + self.l * np.sin(th)
            my = py + self.l * np.cos(th)

            rod.set_data([px, mx], [py, my])
            mass.center = (mx, my)

            if force_arrow[0] is not None:
                force_arrow[0].remove()

            dx = force_scale * u
            force_arrow[0] = ax.arrow(
                x,
                wr + ch + 0.12,
                dx,
                0,
                width=0.01,
                head_width=0.06,
                head_length=min(0.12, abs(dx)),
                length_includes_head=True,
                color="black",
            )

            txt.set_text(f"t={t:.2f}s, u={u:+.2f}")

            t0 = max(0, t - 6)
            j0 = max(0, int(t0 / self.dt))
            ts = np.arange(j0, k + 1) * self.dt
            us = U[j0:k + 1, 0]

            umax_plot = max(np.max(np.abs(us)), 1.0)

            force_ax.set_xlim(t0, t0 + 6)
            force_ax.set_ylim(-1.1 * umax_plot, 1.1 * umax_plot)

            force_line.set_data(ts, us)
            force_dot.set_data([t], [u])

            return cart, w1, w2, rod, mass, txt, force_line, force_dot

        def update(k):
            return draw(X[k], U[k, 0], k)

        ani = FuncAnimation(
            fig,
            update,
            frames=len(X),
            interval=1000 * self.dt,
            blit=False,
            cache_frame_data=False,
        )

        plt.show(block=True)
        return ani
