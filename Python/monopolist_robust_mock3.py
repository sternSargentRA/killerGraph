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


def optimal(sigc):
    """
    Given a sigc (scalar, dtype=float) it returns the optimal value
    function and entropy level.
    """
    Kwo, Pwo, pwo, BigOo, littleoo = Kworst(beta, sigc, fo, A, B, C, Q, R)
    x0 = np.array([[1.], [0.], [0.]])
    Vo = - x0.T.dot(Pwo.dot(x0)) - pwo
    ento = x0.T.dot(BigOo.dot(x0)) + littleoo

    return map(float, [Vo, ento])


def robust(sigc):
    """
    Given a sigc (scalar, dtype=float) it returns the robust value
    function and entropy level.
    """
    Kwr, Pwr, pwr, BigOr, littleor = Kworst(beta, sigc, F9, A, B, C, Q, R)
    x0 = np.array([[1.], [0.], [0.]])
    Vr = - x0.T.dot(Pwr.dot(x0)) - pwr
    entr = x0.T.dot(BigOr.dot(x0)) + littleor

    return map(float, [Vr, entr])


def vf_ent_target(ent_target, ro, bw):
    """
    Compute the value function and entropy levels for a sigma path
    increasing in modulus until it reaches the specified target entropy
    value.

    Parameters
    ==========
    ent_target : scalar
        The target entropy value

    ro : str
        A string specifying whether the robust or optimal solution
        should be used to populate the DataFrame. The only acceptable
        values are 'optimal' and 'robust'.

    bw : str
        A string specifying whether the implied shock path follows best
        or worst assumptions. The only acceptable values are 'best' and
        'worst'.

    Returns
    =======
    df : pd.DataFrame
        A pandas DataFrame containing the value function and entropy
        values up to the ent_target parameter. The index is the
        relevant portion of sig_vec and the columns are 'vf' and
        'ent' for value function and entropy, respectively

    """
    def _populate_df(svec, kind):
        """
        Internal function used to populate DataFrame using artificial
        svec up to the ent_target value
        """
        df = pd.DataFrame(index=svec, columns=['vf', 'ent'])
        if kind == 'robust':
            func = robust
        elif kind == 'optimal':
            func = optimal
        else:
            raise ValueError("Argument 'kind' for function _populate_df"
                             + " must be 'optimal' or 'robust'")

        for i, sigc in enumerate(svec):
            df.ix[sigc] = func(sigc)
            if ent_target - df.ix[sigc, 'ent'] < 0:
                break
        else:
            df = df.dropna()

        return df

    if bw == 'best':
        svec = 0.1 * np.linspace(1e-7, 1000, 1e4)
    elif bw == 'worst':
        svec = -1 * np.linspace(1e-7, 1000, 1e4)
    else:
        raise ValueError("Argument 'bw' must be 'best' or 'worst'")

    return _populate_df(svec, ro)


def vf_ent_sigpath(sigma_vec, skip=10):
    """
    Compute value functions and entropies associated with a stream of
    shocks sigma_vec

    Parameters
    ==========
    sigma_vec : array_like, dtype=float
        The stream of shocks to be used in computing the responses.

    skip : integer, dtype=int
        How many iterations between prints

    Returns
    =======
    data: pd.DataFrame
        The optimal and robust value function and entropy associated
        with the shocks in sigma_vec. The first column of each of these
        arrays is the value function and the second column is the
        entropy.

    """
    N = sigma_vec.size
    data = pd.DataFrame(np.zeros((N, 4)),
                        index=sigma_vec,
                        columns=['opt_vf', 'opt_ent', 'rob_vf', 'rob_ent'])

    for i in xrange(N):
        sigc = sigma_vec[i]
        data.ix[sigc, ['opt_vf', 'opt_ent']] = optimal(sigc)

        data.ix[sigc, ['rob_vf', 'rob_ent']] = robust(sigc)

        if i % skip == 0:
            e_time = time() - start_time
            print(msg.format(num=i, N=N, time=e_time))

    return data


N = 100
sigspace = np.linspace(1e-7, 100, N)

# compute the two worst case shocks and associated value functions and
# entropies affiliated with some other sig called sigc
print("\n" + "#" * 70)
print("Solving original problem using sigma path")
print("#" * 70 + "\n")
print("\nDoing worst case")

worst_df = vf_ent_sigpath(sigma_vec=-sigspace)

# Now do the "best" case
print("\nNow doing best case")
best_df = vf_ent_sigpath(sigma_vec=(0.1 * sigspace))

# Set up figure
plt.figure(1)
plt.ylabel("Value Function")
plt.xlabel("Entropy")
plt.title("Value sets")

# Plot worst case shocks
plt.plot(worst_df['opt_ent'], worst_df['opt_vf'], 'r')
plt.plot(worst_df['rob_ent'], worst_df['rob_vf'], 'b--')

# Plot best case
plt.plot(best_df['opt_ent'], best_df['opt_vf'], 'r')
plt.plot(best_df['rob_ent'], best_df['rob_vf'], 'b--')
plt.show()

ent_target = max(best_df[['opt_ent', 'rob_ent']].max().max(),
                 worst_df[['opt_ent', 'rob_ent']].max().max())

print("\n" + "#" * 70)
print("Now using a grid search method to make all paths go to same entropy")
print("#" * 70 + "\n")

print("\nDoing optimal vf + best-case shock path")
opt_best_df = vf_ent_target(ent_target, 'optimal', 'best')

print("\nDoing optimal vf + worst-case shock path")
opt_worst_df = vf_ent_target(ent_target, 'optimal', 'worst')

print("\nDoing robust vf + best-case shock path")
rob_best_df = vf_ent_target(ent_target, 'robust', 'best')

print("\nDoing robust vf + worst-case shock path")
rob_worst_df = vf_ent_target(ent_target, 'robust', 'worst')

# Generate new plot
fig2 = plt.figure()
ax = fig2.add_subplot(111)

opt_best_df.plot(x='ent', y='vf', style='r', legend=False, ax=ax)
opt_worst_df.plot(x='ent', y='vf', style='r', legend=False, ax=ax)
rob_best_df.plot(x='ent', y='vf', style='b--', legend=False, ax=ax)
rob_worst_df.plot(x='ent', y='vf', style='b--', legend=False, ax=ax)

ax.set_ylabel("Value Function")
ax.set_xlabel("Entropy")
ax.set_title("Value Sets")
plt.savefig('value_sets.eps')

# value_entropy(entrop_target=1572226)
