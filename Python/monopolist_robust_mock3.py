"""
This file mimics monopolist_robust_mock3.m

monopolist with adjustment costs
created as an example in terms of which to cast the `killer graph'
showing robustness. We'll use olrprobust.m to compute key objects
Inverse demand curve:
  p_t = A_0 - A_1 Q_t + d_t
  d_{t+1} = \rho d_t + \sigma_d epsilon_{t+1}
  epsilon_{t+1} is iid N(0,1)
Period return function for monopolist:

  R_t =  p_t Q_t - .5 e (Q_{t+1}-Q_t) - c Q_t   =
  (A_0 - c) Q_t - A_1 Q_t^2 _ d_t Q_t - .5 e Q_t
Objective of the firm:

  E_t \sum_{t=0}^\infty \beta^t R_t
Create linear regulator. State is x_t' = [1, Q_t, d_t]
control is u_t = (Q_{t+1} - Q_t )
The implied state-space matrices are
the A, B, C, Q, R below.
The free parameters are beta, A_0, A_1, rho, sigma_d, beta, e, c
AND the robustness parameter sig \equiv - theta^{-1}

Original Matlab Author: Tom Sargent

Python Authors: Chase Coleman and Spencer Lyon

Date: 2013-08-22 16:10:07
"""
import math
from time import time
import numpy as np
from killergraphfuncs import *
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

#-----------------------------------------------------------------------------#
# Define run control parameters
#-----------------------------------------------------------------------------#
N = 100  # number of values in grid for sigma
skip = 10  # how many iterations to skip before printing update

msg = "Iteration {num} of {N}. Total time elapsed {time}"

start_time = time()

#-----------------------------------------------------------------------------#
# Define parameters to be used
#-----------------------------------------------------------------------------#
A_0 = 100.
A_1 = .5
rho = .9
sigma_d = .05
beta = .95
c = 2.
e = 50.
sig = -10.  # This is the risk sensitives parameter and should be neg when
            # there is fear of model misspecification

#-----------------------------------------------------------------------------#
# Define necessary Matrices
#-----------------------------------------------------------------------------#
Q = np.array([[0, .5*(A_0 - c), 0], [.5*(A_0 - c), -A_1, .5], [0., .5, 0.]])
R = -.5 * e

# Flip signs to make both matrices positive definite since we solve the
# minimization problem
Q = -Q
R = - R

A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., rho]])
B = np.array([[0.], [1.], [0.]])
C = np.array([[0.], [0.], [sigma_d]])
Ad = math.sqrt(beta) * A
Bd = math.sqrt(beta) * B


#-----------------------------------------------------------------------------#
# Evaluating
#-----------------------------------------------------------------------------#

# Compute the optimal rule
fo, po = olrp(beta, A, B, Q, R)

# Compute a robust rule affiliated with sig < 0
sig = -50.  # why are we redefining sig here?
F9, K9, P9, Pt9 = doublex9(Ad, Bd, C, Q, R, sig)

K9 = K9 / math.sqrt(beta)

# F9 is the robust decision rule affiliated with sig = - 1/theta;
# K9 is the worst case shock coefficient matrix affiliated with that rule and that sig.

# compute nonstochastic steady state

# xs is the steady state for [1, Q, d] under the robust decision rule F
# under the APPROXIMATING model


# Check the positive definiteness of the worst-case covariance matrix to
# assure that theta exceeds the breakdown point

check = eye(P9.shape[0]) + sig * np.dot(C.T, P9.dot(C))

checkfinal = eig(check)[0]

# Check the above ^^
if (checkfinal < 0).any():
    raise ValueError("Theta doesn't exceed breakdown point. Rechoose params.")


def value_entropy(sigma_vec=None,  entrop_bound=None,  skip=10):
    """
    Compute value functions and entropies associated with a stream of
    shocks sigma_vec

    Parameters
    ==========
    sigma_vec : array_like, dtype=float
        The stream of shocks to be used in computing the responses.
    entrop_bound : scalar, dtype=float
        The maximum level of entropy that we want
    skip : integer, dtype=int
        How many iterations between prints

    Returns
    =======
    opt, robust: array_like, dtype=float, shape=(sigma_vec.size, 2)
        The optimal and robust value function and entropy associated
        with the shocks in sigma_vec. The first column of each of these
        arrays is the value function and the second column is the
        entropy.

    """
    def calc_func(sigc):
        """
        Given a sigc (scalar, dtype=float) it returns Vo, Vr, ento, entr
        that are used to calculate value func and entropy for sigc
        """
        Kwo, Pwo, pwo, BigOo, littleoo = Kworst(beta, sigc, fo, A, B, C, Q, R)
        Kwr, Pwr, pwr, BigOr, littleor = Kworst(beta, sigc, F9, A, B, C, Q, R)

        # Now compute vf and entropies evaluated at init state x0 = [1, 0, 0]'

        x0 = np.array([[1.], [0.], [0.]])

        Vo = - x0.T.dot(Pwo.dot(x0)) - pwo
        Vr = - x0.T.dot(Pwr.dot(x0)) - pwr

        ento = x0.T.dot(BigOo.dot(x0)) + littleoo
        entr = x0.T.dot(BigOr.dot(x0)) + littleor

        return Vo, Vr, ento, entr

    if sigma_vec is not None and entrop_bound is not None:
        raise ValueError('Cannot define both sigma_vec and \
                          entropy_bound.  Try again.')

    # If we have entrop_bound then use it to calculate the desired results
    if entrop_bound:
        print('entrop_bound')

    else:
        N = sigma_vec.size
        data = pd.DataFrame(np.zeros((N, 4)),
                            index=sigma_vec,
                            columns=['opt_vf', 'opt_ent', 'rob_vf', 'rob_ent'])

        for i in xrange(N):
            sigc = sigma_vec[i]

            Vo, Vr, ento, entr = calc_func(sigc)

            data['opt_vf'][i] = Vo
            data['opt_ent'][i] = ento

            data['rob_vf'][i] = Vr
            data['rob_ent'][i] = entr

        if i % skip == 0:
            e_time = time() - start_time
            print(msg.format(num=i, N=N, time=e_time))

    return data


N = 100
sigspace = np.linspace(1e-7, 100, N)

# compute the two worst case shocks and associated value functions and
# entropies affiliated with some other sig called sigc
worst_dat = value_entropy(sigma_vec=-sigspace)

# Now do the "optimistic" case
print("\n" + "#" * 70 + "\nMoving on to optimistic case\n" + "#" * 70 + "\n")
optimistic_dat = value_entropy(sigma_vec=(0.1 * sigspace))

# Set up figure
plt.figure(1)
plt.ylabel("Value Function")
plt.xlabel("Entropy")
plt.title("Value sets")

# Plot worst case shocks
plt.plot(worst_dat['opt_ent'], worst_dat['opt_vf'], 'r')
plt.plot(worst_dat['rob_ent'], worst_dat['rob_vf'], 'b--')

# Plot optimistic case
plt.plot(optimistic_dat['opt_ent'], optimistic_dat['opt_vf'], 'r')
plt.plot(optimistic_dat['rob_ent'], optimistic_dat['rob_vf'], 'b--')
plt.show()
