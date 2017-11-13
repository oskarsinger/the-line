import numpy as np

def get_simple_planted_clique(d, rho=0.8):

    I = np.eye(d)

    I[1,0] = rho
    I[0,1] = rho

    return I
