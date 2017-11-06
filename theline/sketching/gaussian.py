import numpy as np

class GaussianSketcher:

    def __init__(self, n, m):

        self.n = n
        self.m = m

        self.S = np.random.randn(n, m)

    def get_sketched(self, A):

        return np.dot(A, self.S)

    def get_unsketched(self, A):

        return np.dot(self.S, A)
