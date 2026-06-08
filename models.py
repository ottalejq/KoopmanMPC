import numpy as np
import torch
import osqp
from scipy import sparse





class BaseMPC:
    def predict(self, X, U):
        # Allow single state input
        X = X[None] if X.ndim == 1 else X
        m, _ = U.shape

        # Normalize state and input
        X = (X - self.mx) / self.sx
        U = (U - self.mu) / self.su

        # Initial lifted state
        Z = self.lift_np(X)
        Xn = np.empty((m, self.p))

        # Roll out dynamics forward
        for k, u in enumerate(U):
            H = np.c_[Z, Z * u, u[None]]
            Z = H @ self.W
            Xn[k] = Z[:, :self.p]

        # Convert back to original coordinates
        return Xn * self.sx + self.mx

    def local_model(self, x, X_ref, U_ref):
        # Linearize along reference trajectory
        H, _ = U_ref.shape

        A = np.empty((H, self.p, self.p))
        B = np.empty((H, self.p, 1))

        for k in range(H):
            A[k], B[k] = self.linearize(X_ref[k], U_ref[k])

        return A, B

    def LQR(self, x, X_ref, U_ref):
        # Time-varying linear model
        A, B = self.local_model(x, X_ref, U_ref)

        H, _ = U_ref.shape

        # Backward Riccati recursion
        P = self.Q.copy()
        K = np.empty((H, 1, self.p))

        for k in range(H - 1, -1, -1):
            S = self.R + B[k].T @ P @ B[k]
            K[k] = np.linalg.solve(S, B[k].T @ P @ A[k])
            P = self.Q + A[k].T @ P @ (A[k] - B[k] @ K[k])

        # Apply first feedback law
        return U_ref[0] - K[0] @ (x - X_ref[0])

    def MPC(self, x, X_ref, U_ref, u_bounds=(-20, 20)):
        # Linearize along reference trajectory
        A, B = self.local_model(x, X_ref, U_ref)

        H, _ = U_ref.shape
        nx, nu = self.p, 1

        # Prediction matrices
        M = np.zeros((H * nx, nx))
        S = np.zeros((H * nx, H * nu))

        Phi = np.eye(nx)

        for i in range(H):
            Phi = A[i] @ Phi
            M[i * nx:(i + 1) * nx] = Phi

            G = B[i]
            S[i * nx:(i + 1) * nx, i * nu:(i + 1) * nu] = G

            for j in range(i - 1, -1, -1):
                G = A[j + 1] @ G
                S[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = G

        # Block cost matrices
        Qbar = np.kron(np.eye(H), self.Q)
        Rbar = np.kron(np.eye(H), self.R)

        # Initial state error
        dx = x - X_ref[0]

        # Quadratic program cost
        P = S.T @ Qbar @ S + Rbar
        q = S.T @ Qbar @ (M @ dx)

        # Input constraints
        umin, umax = u_bounds

        C = np.eye(H * nu)
        l = np.tile(umin, H) - U_ref[:H].squeeze()
        u = np.tile(umax, H) - U_ref[:H].squeeze()

        # Solve QP
        prob = osqp.OSQP()
        prob.setup(
            P=sparse.csc_matrix(P),
            q=q,
            A=sparse.csc_matrix(C),
            l=l,
            u=u,
            verbose=False,
        )

        res = prob.solve()

        # Apply first optimal control move
        return U_ref[0] + res.x[:nu]







class RFFMPC(BaseMPC):
    def __init__(self, D, gamma, lam, Q, R, seed=42):
        # Model and controller parameters
        self.rng = np.random.default_rng(seed)
        self.D = D
        self.gamma = gamma
        self.lam = lam
        self.Q = Q
        self.R = R
        self.eps = 1e-8

    def fit(self, X, Xn, U):
        # State dimension
        _, self.p = X.shape

        # Compute normalization statistics
        self.mx = X.mean(axis=0)
        self.sx = np.sqrt(X.var(axis=0) + self.eps)

        self.mu = U.mean(axis=0)
        self.su = np.sqrt(U.var(axis=0) + self.eps)

        # Normalize training data
        X = (X - self.mx) / self.sx
        Xn = (Xn - self.mx) / self.sx
        U = (U - self.mu) / self.su

        # Sample random Fourier features
        self.omega = self.rng.normal(
            0.0,
            np.sqrt(2.0 * self.gamma),
            size=(self.p, self.D),
        )

        self.beta = self.rng.uniform(
            0.0,
            2.0 * np.pi,
            size=self.D,
        )

        # Lift current and next states
        Z = self.lift_np(X)
        Zn = self.lift_np(Xn)

        # Build Koopman regression matrix
        H = np.c_[Z, Z * U, U]

        # Solve ridge regression
        n = H.shape[1]
        self.W = np.linalg.solve(
            H.T @ H + self.lam * np.eye(n),
            H.T @ Zn,
        )

        return self

    def lift_np(self, X):
        # Random Fourier feature map
        X = X[None] if X.ndim == 1 else X
        c = np.sqrt(2.0 / self.D)

        return np.c_[
            X,
            c * np.cos(X @ self.omega + self.beta),
        ]

    def linearize(self, x_ref, u_ref):
        # Normalize operating point
        x = (x_ref - self.mx) / self.sx
        u = (u_ref - self.mu) / self.su

        # Evaluate lifted state
        c = np.sqrt(2.0 / self.D)
        a = x @ self.omega + self.beta
        z = np.r_[x, c * np.cos(a)]

        # Jacobian of lifting map
        dzdx = np.r_[
            np.eye(self.p),
            -c * np.sin(a)[:, None] * self.omega.T,
        ]

        # Jacobian with respect to state
        dhdx = np.r_[
            dzdx,
            u * dzdx,
            np.zeros((1, self.p)),
        ]

        # Jacobian with respect to input
        dhdu = np.r_[
            np.zeros((len(z), 1)),
            z[:, None],
            np.eye(1),
        ]

        # Convert lifted Jacobians to state-space form
        W = self.W[:, :self.p]

        A = self.sx[:, None] * (W.T @ dhdx) / self.sx[None, :]
        B = self.sx[:, None] * (W.T @ dhdu) / self.su[None, :]

        return A, B
    











class LearnedRFFMPC(BaseMPC):
    def __init__(self, D, gamma, lam, Q, R, lr=1.0, wd=1e-2, epochs=200, seed=42):
        # Model, training, and controller parameters
        self.rng = np.random.default_rng(seed)
        self.D = D
        self.gamma = gamma
        self.lam = lam
        self.Q = Q
        self.R = R
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.eps = 1e-8
        torch.manual_seed(seed)

    def fit(self, X, Xn, U):
        # State dimension
        _, self.p = X.shape

        # Compute normalization statistics
        self.mx = X.mean(axis=0)
        self.sx = np.sqrt(X.var(axis=0) + self.eps)

        self.mu = U.mean(axis=0)
        self.su = np.sqrt(U.var(axis=0) + self.eps)

        # Normalize training data
        X = (X - self.mx) / self.sx
        Xn = (Xn - self.mx) / self.sx
        U = (U - self.mu) / self.su

        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        Xn = torch.tensor(Xn, dtype=torch.float32)
        U = torch.tensor(U, dtype=torch.float32)

        # Initialize learnable cosine features
        self.omega = torch.nn.Parameter(torch.tensor(
            self.rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(self.p, self.D)),
            dtype=torch.float32,
        ))

        self.beta = torch.nn.Parameter(torch.tensor(
            self.rng.uniform(0.0, 2.0 * np.pi, size=self.D),
            dtype=torch.float32,
        ))

        opt = torch.optim.AdamW(
            [self.omega, self.beta],
            lr=self.lr,
            weight_decay=self.wd
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.epochs
        )

        for epoch in range(self.epochs):
            opt.zero_grad()

            Z = self.lift_torch(X)
            Zn = self.lift_torch(Xn)

            H = torch.cat([Z, Z * U, U], dim=1)
            I = torch.eye(H.shape[1], dtype=H.dtype, device=H.device)

            W = torch.linalg.solve(
                H.T @ H + self.lam * I,
                H.T @ Zn,
            )

            loss = ((H @ W - Zn) ** 2).mean()

            loss.backward()
            opt.step()
            scheduler.step()

            if epoch % 10 == 0:
                print(
                    f"epoch {epoch} | "
                    f"loss {loss.item():.6e} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e}"
                )

        # Refit Koopman operator with final features
        with torch.no_grad():
            Z = self.lift_torch(X)
            Zn = self.lift_torch(Xn)

            H = torch.cat([Z, Z * U, U], dim=1)
            I = torch.eye(H.shape[1], dtype=H.dtype, device=H.device)

            W = torch.linalg.solve(
                H.T @ H + self.lam * I,
                H.T @ Zn,
            )

        # Store learned parameters as numpy arrays
        self.omega = self.omega.detach().numpy()
        self.beta = self.beta.detach().numpy()
        self.W = W.detach().numpy()

        return self

    def lift_torch(self, X):
        # Torch version used during training
        c = np.sqrt(2.0 / self.D)

        return torch.cat([
            X,
            c * torch.cos(X @ self.omega + self.beta),
        ], dim=1)

    def lift_np(self, X):
        # Numpy version used during prediction/control
        X = X[None] if X.ndim == 1 else X
        c = np.sqrt(2.0 / self.D)

        return np.c_[
            X,
            c * np.cos(X @ self.omega + self.beta),
        ]

    def linearize(self, x_ref, u_ref):
        # Normalize operating point
        x = (x_ref - self.mx) / self.sx
        u = (u_ref - self.mu) / self.su

        # Evaluate learned lifting map
        c = np.sqrt(2.0 / self.D)
        a = x @ self.omega + self.beta
        z = np.r_[x, c * np.cos(a)]

        # Jacobian of lifting map
        dzdx = np.r_[
            np.eye(self.p),
            -c * np.sin(a)[:, None] * self.omega.T,
        ]

        # Jacobian with respect to state
        dhdx = np.r_[
            dzdx,
            u * dzdx,
            np.zeros((1, self.p)),
        ]

        # Jacobian with respect to input
        dhdu = np.r_[
            np.zeros((len(z), 1)),
            z[:, None],
            np.eye(1),
        ]

        # Convert to state-space model
        W = self.W[:, :self.p]

        A = self.sx[:, None] * (W.T @ dhdx) / self.sx[None, :]
        B = self.sx[:, None] * (W.T @ dhdu) / self.su[None, :]

        return A, B
    

















class DeepRFFMPC(BaseMPC):
    def __init__(self, D, gamma, lam, Q, R, lr=1.0, wd=1e-2, epochs=200, seed=42):
        # Two-layer learned lifting model
        self.rng = np.random.default_rng(seed)
        self.D = D
        self.gamma = gamma
        self.lam = lam
        self.Q = Q
        self.R = R
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.eps = 1e-8
        torch.manual_seed(seed)

    def fit(self, X, Xn, U):
        # State dimension
        _, self.p = X.shape

        # Compute normalization statistics
        self.mx = X.mean(axis=0)
        self.sx = np.sqrt(X.var(axis=0) + self.eps)

        self.mu = U.mean(axis=0)
        self.su = np.sqrt(U.var(axis=0) + self.eps)

        # Normalize training data
        X = (X - self.mx) / self.sx
        Xn = (Xn - self.mx) / self.sx
        U = (U - self.mu) / self.su

        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        Xn = torch.tensor(Xn, dtype=torch.float32)
        U = torch.tensor(U, dtype=torch.float32)

        # Initialize first cosine layer
        self.omega1 = torch.nn.Parameter(torch.tensor(
            self.rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(self.p, self.D)),
            dtype=torch.float32,
        ))

        self.beta1 = torch.nn.Parameter(torch.tensor(
            self.rng.uniform(0.0, 2.0 * np.pi, size=self.D),
            dtype=torch.float32,
        ))

        # Initialize second cosine layer
        self.omega2 = torch.nn.Parameter(torch.tensor(
            self.rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(self.D, self.D)),
            dtype=torch.float32,
        ))

        self.beta2 = torch.nn.Parameter(torch.tensor(
            self.rng.uniform(0.0, 2.0 * np.pi, size=self.D),
            dtype=torch.float32,
        ))


        opt = torch.optim.AdamW(
            [self.omega1, self.beta1, self.omega2, self.beta2],
            lr=self.lr,
            weight_decay=self.wd,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.epochs
        )
        
        for epoch in range(self.epochs):
            opt.zero_grad()

            Z = self.lift_torch(X)
            Zn = self.lift_torch(Xn)

            H = torch.cat([Z, Z * U, U], dim=1)
            I = torch.eye(H.shape[1], dtype=H.dtype, device=H.device)

            W = torch.linalg.solve(
                H.T @ H + self.lam * I,
                H.T @ Zn,
            )

            loss = ((H @ W - Zn) ** 2).mean()

            loss.backward()
            opt.step()
            scheduler.step()

            if epoch % 10 == 0:
                print(
                    f"epoch {epoch} | "
                    f"loss {loss.item():.6e} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e}"
                )

        # Refit Koopman operator with final features
        with torch.no_grad():
            Z = self.lift_torch(X)
            Zn = self.lift_torch(Xn)

            H = torch.cat([Z, Z * U, U], dim=1)
            I = torch.eye(H.shape[1], dtype=H.dtype, device=H.device)

            W = torch.linalg.solve(
                H.T @ H + self.lam * I,
                H.T @ Zn,
            )

        self.omega1 = self.omega1.detach().numpy()
        self.beta1 = self.beta1.detach().numpy()
        self.omega2 = self.omega2.detach().numpy()
        self.beta2 = self.beta2.detach().numpy()
        self.W = W.detach().numpy()

        return self


    def lift_torch(self, X):
        # Torch version used during training
        c = np.sqrt(2.0 / self.D)

        # First cosine layer
        h1 = c * torch.cos(X @ self.omega1 + self.beta1)

        # Second cosine layer
        h2 = c * torch.cos(h1 @ self.omega2 + self.beta2)

        # Full lifted state
        return torch.cat([X, h1, h2], dim=1)

    def lift_np(self, X):
        # Numpy version used during prediction/control
        X = X[None] if X.ndim == 1 else X
        c = np.sqrt(2.0 / self.D)

        # First cosine layer
        h1 = c * np.cos(X @ self.omega1 + self.beta1)

        # Second cosine layer
        h2 = c * np.cos(h1 @ self.omega2 + self.beta2)

        return np.c_[X, h1, h2]

    def linearize(self, x_ref, u_ref):
        # Normalize operating point
        x = (x_ref - self.mx) / self.sx
        u = (u_ref - self.mu) / self.su

        # Forward pass through lifting layers
        c = np.sqrt(2.0 / self.D)

        a1 = x @ self.omega1 + self.beta1
        h1 = c * np.cos(a1)

        a2 = h1 @ self.omega2 + self.beta2
        h2 = c * np.cos(a2)

        z = np.r_[x, h1, h2]

        # Chain-rule Jacobian of first layer
        dh1dx = -c * np.sin(a1)[:, None] * self.omega1.T

        # Chain-rule Jacobian of second layer
        dh2dh1 = -c * np.sin(a2)[:, None] * self.omega2.T
        dh2dx = dh2dh1 @ dh1dx

        # Jacobian of full lifted state
        dzdx = np.r_[
            np.eye(self.p),
            dh1dx,
            dh2dx,
        ]

        # Jacobian with respect to state
        dhdx = np.r_[
            dzdx,
            u * dzdx,
            np.zeros((1, self.p)),
        ]

        # Jacobian with respect to input
        dhdu = np.r_[
            np.zeros((len(z), 1)),
            z[:, None],
            np.eye(1),
        ]

        # Convert to state-space model
        W = self.W[:, :self.p]

        A = self.sx[:, None] * (W.T @ dhdx) / self.sx[None, :]
        B = self.sx[:, None] * (W.T @ dhdu) / self.su[None, :]

        return A, B