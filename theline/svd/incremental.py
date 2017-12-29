import numpy as np

# TODO: cite Markos 2016 paper
class RowIncrementalSVD:

    def __init__(self, k):

        self.k = k
        self.U = None
        self.Sigma = None
        self.VT = None

        self.num_rounds = 0

    def get_update(self, A):

        if self.num_rounds == 0:
            self.m = A.shape[1]
            (self.U, self.Sigma, self.VT) = np.linalg.svd(A)
        else:
            (U_new, Sigma_new, VT_new) = np.linalg.svd(A)
            L = np.dot(A, self.VT)
            H = A - np.dot(L, self.VT.T)
            V_H = np.linalg.qr(H)[0]
            R = self._get_R(L, H, V_H)

            self._set_new_svd(R, V_H)

        return (self.U, self.Sigma, self.VT)

    def _set_new_svd(self, R, V_H):

        (U_R, Sigma_R, VT_R) = np.linalg.svd(R)

        diff = U_R.shape[0] - self.U.shape[1]
        rows = self.U.shape[0] + diff
        cols = self.U.shape[1] + diff
        padded = np.zeros((rows, cols))

        padded[:self.U.shape[0],:self.U.shape[1]] += self.U
        padded[self.U.shape[0]:,self.U.shape[1]:] += np.eye(diff)

        self.U = np.dot(padded, U_R)
        self.Sigma = Sigma_R

        self.VT = np.dot(VT_R, np.vstack([self.VT, V_H]))

    def _get_R(self, L, H, V_H):

        R_rows = self.m + L.shape[0]
        R_cols = self.m + self.VT.shape[1]
        R = np.zeros((R_rows, R_cols))
        R[:self.m,:self.m] += self.Sigma
        R[self.m:,:self.m] += L
        R[self.m:,self.m:] += np.dot(H, V_H.T)

        return R

# TODO: cite Baker 2008 paper
class ColumnIncrementalSVD:

    def __init__(self, k):

        self.k = k

        self.Q = None
        self.B = None
        self.W = None
        self.m = None
        self.l = 0
        self.num_rounds = 0

    def get_update(self, A):

        Q_bar = None
        B_bar = None
        W_bar = None
        lt = A.shape[1]

        if self.num_rounds == 0:
            self.m = A.shape[0]
            (Q_bar, B) = np.linalg.qr(A)
            B_bar = np.diag(B)
            W_bar = np.eye(lt)
        else:
            (Q_hat, B_hat) = self._get_QB_hat(A)
            W_hat = self._get_W_hat(lt)
            (G_u, B_bar, G_vT) = np.linalg.svd(
                B_hat, full_matrices=False)
            Q_bar = np.dot(Q_hat, G_u)
            W_bar = np.dot(W_hat, G_vT.T)
        
        self.B = B_bar[:self.k]
        self.Q = Q_bar[:,:self.k]
        self.W = W_bar[:,:self.k]
        self.l += lt
        self.num_rounds += 1

        return (self.Q, self.B, self.W.T)

    def _get_W_hat(self, l):

        (Wn, Wm) = self.W.shape
        W_hat = np.zeros(
            (Wn + l, Wm + l))
        W_hat[:Wn,:Wm] += self.W
        W_hat[Wn:,Wm:] += np.eye(l)

        return W_hat

    def _get_QB_hat(self, A):

        lt = A.shape[1]
        kt = min(self.k, self.l)
        C = np.dot(self.Q.T, A)
        pre_QR = A - np.dot(self.Q, C)
        (Q_perp, B_perp) = np.linalg.qr(pre_QR)
        Q_hat = np.hstack([self.Q, Q_perp])
        B_hat = np.zeros((kt + lt, kt + lt))
        B_hat[:kt,:kt] += np.diag(self.B)
        B_hat[:kt,kt:] += C
        B_hat[kt:,kt:] += B_perp

        return (Q_hat, B_hat)
