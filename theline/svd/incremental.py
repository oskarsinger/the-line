import numpy as np

from theline.utils import get_multi_dot
from theline.svd import get_multiplied_svd as get_ms

# TODO: take care of centering

# TODO: cite Markos 2016 paper
class RowIncrementalSVD:

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
        lt = A.shape[0]

        if self.num_rounds == 0:
            self.m = A.shape[1]
        else:
            pass

    def _get_QB_hat(self, A):
        pass

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
            (Q_bar, B) = np.thelineg.qr(A)
            B_bar = np.diag(B)
            W_bar = np.eye(lt)
        else:
            (Q_hat, B_hat) = self._get_QB_hat(A)
            W_hat = self._get_W_hat(lt)
            (G_u, B_bar, G_vT) = np.thelineg.svd(
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
        (Q_perp, B_perp) = np.thelineg.qr(pre_QR)
        Q_hat = np.hstack([self.Q, Q_perp])
        B_hat = np.zeros((kt + lt, kt + lt))
        B_hat[:kt,:kt] += np.diag(self.B)
        B_hat[:kt,kt:] += C
        B_hat[kt:,kt:] += B_perp

        return (Q_hat, B_hat)
