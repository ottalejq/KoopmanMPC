import numpy as np


class SinCosEncoder:
    """Encode an angular state using sine and cosine."""

    def __init__(self, model, i=2):
        self.model = model
        self.i = i
        self.p = 4

    @staticmethod
    def convert_Q(Q, i=2):
        # Expand the angular cost to the sine-cosine representation.
        Q_enc = np.zeros((Q.shape[0] + 1, Q.shape[1] + 1))

        Q_enc[:i, :i] = Q[:i, :i]
        Q_enc[i, i] = Q[i, i]
        Q_enc[i + 1, i + 1] = Q[i, i]
        Q_enc[i + 2:, i + 2:] = Q[i + 1:, i + 1:]

        return Q_enc

    def enc(self, X):
        # Replace the angle with its sine and cosine.
        vec = X.ndim == 1
        X = X[None, :] if vec else X

        X_enc = np.c_[
            X[:, :self.i],
            np.sin(X[:, self.i]),
            np.cos(X[:, self.i]),
            X[:, self.i + 1:],
        ]

        return X_enc[0] if vec else X_enc

    def dec(self, X):
        # Recover the angle using atan2.
        if X.ndim == 1:
            return np.r_[
                X[:self.i],
                np.arctan2(X[self.i], X[self.i + 1]),
                X[self.i + 2:],
            ]

        return np.concatenate([
            X[..., :self.i],
            np.arctan2(X[..., self.i], X[..., self.i + 1])[..., None],
            X[..., self.i + 2:],
        ], axis=-1)

    def fit(self, X, Xn, U):
        # Train the wrapped model in encoded coordinates.
        self.model.fit(
            self.enc(X),
            self.enc(Xn),
            U,
        )
        return self

    def init_global_control(self, Q, R, Qt, N):
        self.model.init_global_control(
            Q=self.convert_Q(Q),
            R=R,
            Qt=self.convert_Q(Qt),
            N=N,
        )

    def global_control(self, x, **kwargs):
        out = self.model.global_control(x=self.enc(x), **kwargs)

        # Decode predicted states when requested.
        if kwargs.get("return_state", False):
            u, x_next_enc = out
            return u, self.dec(x_next_enc)

        return out

    def init_local_control(self, Q, R, Rd, N):
        self.model.init_local_control(
            Q=self.convert_Q(Q),
            R=R,
            Rd=Rd,
            N=N,
        )

    def local_control(self, x, X_ref, U_ref, W_ref, **kwargs):
        out = self.model.local_control(
            x=self.enc(x),
            X_ref=self.enc(X_ref),
            U_ref=U_ref,
            W_ref=W_ref,
            **kwargs,
        )

        # Decode predicted states when requested.
        if kwargs.get("return_state", False):
            u, x_next_enc = out
            return u, self.dec(x_next_enc)

        return out

    def rollout_forward(self, x, U, **kwargs):
        # Roll out forward in encoded coordinates.
        return self.dec(
            self.model.rollout_forward(self.enc(x), U, **kwargs)
        )

    def rollout_backward(self, x, U, **kwargs):
        # Roll out backward in encoded coordinates.
        return self.dec(
            self.model.rollout_backward(self.enc(x), U, **kwargs)
        )