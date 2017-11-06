import numpy as np

class GaussianSketcher:

    def __init__(self, n, m=None):

        self.n = n

        if m is None:
            m = int(0.1 * self.n)

        self.m = m

        self.S = np.random.randn(self.n, self.m)

    def get_sketched(self, A):

        return np.dot(A, self.S)

    def get_unsketched(self, A):

        return np.dot(self.S, A)
