import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
from scipy.linalg import expm, solve_discrete_are


class CartPole:
    def __init__(
        self,
        T=10,
        dt=1/50,
        n_substeps=5,
        M=1.0,
        m=1.0,
        l=1.0,
        g=9.81
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


    def sample_reference(
        self,
        sigma_X,
        sigma_U,
        n_samples_per_step,
    ):
        print('Generating training samples.')

        X, Xn, U = [], [], []

        for x_ref, u_ref, K in zip(self.X_ref, self.U_ref, self.K):
            dx = np.random.multivariate_normal(
                mean=np.zeros(x_ref.shape),
                cov=sigma_X**2,
                size=n_samples_per_step,
            )

            du = np.random.normal(
                loc=0.0,
                scale=sigma_U,
                size=n_samples_per_step,
            )

            x = x_ref + dx
            u = u_ref - dx @ K.reshape(-1) + du
            xn = self.step(x, u)

            X.append(x)
            Xn.append(xn)
            U.append(u)

        return (
            np.vstack(X),
            np.vstack(Xn),
            np.concatenate(U),
        )

    def generate_process_noise(self, sigma_X):
        self.process_noise = np.random.multivariate_normal(
            mean=np.zeros(4),
            cov=sigma_X**2,
            size=self.N,
        )
        return self

    def simulate(self, policy, H, animate=True):
        x = self.X0

        X = np.zeros((self.N - H, 4))
        Xn = np.zeros((self.N - H, 4))
        U = np.zeros((self.N - H, 1))

        for k in range(self.N - H):
            print(
                f"\rSimulating | {(k + 1) / (self.N - H):.2f}",
                end="",
                flush=True,
            )

            X_ref = self.X_ref[k: k + H]
            U_ref = self.U_ref[k: k + H]

            u = policy(x, X_ref, U_ref)

            xn = self.step(x, u)

            xn += self.process_noise[k]

            X[k] = x
            Xn[k] = xn
            U[k, 0] = u

            x = xn

        print('')

        if animate:
            self.ani = self.animate(X, U)

        return X, Xn, U


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

        plt.show()
        return ani