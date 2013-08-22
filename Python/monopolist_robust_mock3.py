# This file mimics monopolist_robust_mock3.m

import numpy as np
import math


#-----------------------------------------------------------------------------#
# Description
#-----------------------------------------------------------------------------#
# monopolist with adjustment costs
# created as an example in terms of which to cast the `killer graph'
# showing robustness. We'll use olrprobust.m to compute key objects
# Inverse demand curve:
#   p_t = A_0 - A_1 Q_t + d_t
#   d_{t+1} = \rho d_t + \sigma_d epsilon_{t+1}
#   epsilon_{t+1} is iid N(0,1)
# Period return function for monopolist:
#
#   R_t =  p_t Q_t - .5 e (Q_{t+1}-Q_t) - c Q_t   =
#   (A_0 - c) Q_t - A_1 Q_t^2 _ d_t Q_t - .5 e Q_t
# Objective of the firm:
#
#   E_t \sum_{t=0}^\infty \beta^t R_t
# Create linear regulator. State is x_t' = [1, Q_t, d_t]
# control is u_t = (Q_{t+1} - Q_t )
# The implied state-space matrices are
# the A, B, C, Q, R below.
# The free parameters are beta, A_0, A_1, rho, sigma_d, beta, e, c
# AND the robustness parameter sig \equiv - theta^{-1}
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Define parameters to be used
#-----------------------------------------------------------------------------#
A_0 = 100.
A_1 = 1.
rho = .9
sigma_d = .05
beta = .95
c = 2.
e = 5.
sig = -10. # This is the risk sensitives parameter and should be neg when there
           # is fear of model misspecification
#-----------------------------------------------------------------------------#


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
Ad = np.sqrt(beta) * A
Bd = np.sqrt(beta) * B


#-----------------------------------------------------------------------------#
#
#-----------------------------------------------------------------------------#
