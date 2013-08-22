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
if checkfinal.any() < 0:
    raise ValueError('Theta does not exceed breakdown point. Rechoose parameters.')

#-----------------------------------------------------------------------------#
# Now compute the two worst case shocks and associated value functions
# and entropies affiliated with some other sig called sigc
#-----------------------------------------------------------------------------#

N = 100

Xopt = np.zeros((N, 2))
Xrobust = np.zeros((N, 2))

sigspace = np.linspace(1e-7, 100, N)

for i in xrange(N):
    sigc = - sigspace[i]

    Kwo, Pwo, pwo, BigOo, littleoo = Kworst(beta, sigc, fo, A, B, C, Q, R)
    Kwr, Pwr, pwr, BigOr, littleor = Kworst(beta, sigc, F9, A, B, C, Q, R)

    # Now compute vf and entropies evaluated at init state x0 = [1, 0, 0]'

    x0 = np.array([[1.], [0.], [0.]])

    Vo = - x0.T.dot(Pwo.dot(x0)) - pwo
    Vr = - x0.T.dot(Pwr.dot(x0)) - pwr

    ento = x0.T.dot(BigOo.dot(x0)) + littleoo
    entr = x0.T.dot(BigOr.dot(x0)) + littleor

    Xopt[i, 0] = Vo
    Xopt[i, 1] = ento

    Xrobust[i, 0] = Vr
    Xrobust[i, 1] = entr

    if i % skip == 0:
        e_time = time() - start_time
        print(msg.format(num=i, N=N, time=e_time))

plt.figure(1)
plt.plot(Xopt[:, 1], Xopt[:, 0], 'r')
plt.plot(Xrobust[:, 1], Xrobust[:, 0], 'b--')

print("\n" + "#" * 70 + "\nMoving on to optimistic case\n" + "#" * 70 + "\n")
# Now do the "optimistic" shock calculations
Yopt = np.zeros((N, 2))
Yrobust = np.zeros((N, 2))

for i in xrange(N):
    sigc = .1 * sigspace[i]

    Kwo, Pwo, pwo, BigOo, littleoo = Kworst(beta, sigc, fo, A, B, C, Q, R)
    Kwr, Pwr, pwr, BigOr, littleor = Kworst(beta, sigc, F9, A, B, C, Q, R)

    # Now compute vf and entropies evaluated at init state x0 = [1, 0, 0]'

    x0 = np.array([[1.], [0.], [0.]])

    Vo = - x0.T.dot(Pwo.dot(x0)) - pwo
    Vr = - x0.T.dot(Pwr.dot(x0)) - pwr

    ento = x0.T.dot(BigOo.dot(x0)) + littleoo
    entr = x0.T.dot(BigOr.dot(x0)) + littleor

    Yopt[i, 0] = Vo
    Yopt[i, 1] = ento

    Yrobust[i, 0] = Vr
    Yrobust[i, 1] = entr

    if i % skip == 0:
        e_time = time() - start_time
        print(msg.format(num=i, N=N, time=e_time))

plt.plot(Yopt[:, 1], Yopt[:, 0], 'r')
plt.plot(Yrobust[:, 1], Yrobust[:, 0], 'b--')
plt.show()
