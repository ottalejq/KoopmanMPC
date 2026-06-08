import numpy as np

class SinCosEncoder:
    def __init__(self, model, i=2):
        self.model, self.i = model, i

    @staticmethod
    def convert_Q(Q, i=2):
        return np.diag(np.insert(
            np.diag(Q), i + 1, Q[i, i])
        )

    def enc(self, X):
        vec = X.ndim == 1
        X = X[None] if vec else X

        X_enc = np.c_[
            X[:, :self.i],
            np.sin(X[:, self.i]),
            np.cos(X[:, self.i]),
            X[:, self.i + 1:]
        ]

        return X_enc[0] if vec else X_enc

    def dec(self, X):
        vec = X.ndim == 1
        X = X[None] if vec else X

        X_dec = np.c_[
            X[:, :self.i],
            np.arctan2(
                X[:, self.i], 
                X[:, self.i + 1]
            ),
            X[:, self.i + 2:]
        ]

        return X_dec[0] if vec else X_dec
    
    def fit(self, X, Xn, U):
        self.model.fit(self.enc(X), self.enc(Xn), U)
        return self

    def predict(self, X, U):
        return self.dec(self.model.predict(self.enc(X), U))

    def LQR(self, x, X_ref, U_ref):
        return self.model.LQR(self.enc(x), self.enc(X_ref), U_ref)

    def MPC(self, x, X_ref, U_ref):
        return self.model.MPC(self.enc(x), self.enc(X_ref), U_ref)



