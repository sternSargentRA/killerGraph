"""
This file holds the functions needed for making a killerGraph

Authors: Chase Coleman, Spencer Lyon

Date: 2013-08-22 10:09:04
"""
from __future__ import division, print_function
from math import sqrt
import numpy as np
from numpy import dot, eye
from scipy.linalg import solve, eig, norm, inv


def doublej(a1, b1, max_it=50):
    """
    Computes the infinite sum V given by

    .. math::

        V = \sum_{j=0}^{\infty} a1^j b1 a1^j'

    where a1 and b1 are each (n X n) matrices with eigenvalues whose
    moduli are bounded by unity and b1 is an (n X n) matrix.

    V is computed by using the following 'doubling algorithm'. We
    iterate to convergence on V(j) on the following recursions for
    j = 1, 2, ... starting from V(0) = b1:

    ..math::

        a1_j = a1_{j-1} a1_{j-1}
        V_j = V_{j-1} + A_{j-1} V_{j-1} a_{j-1}'

    The limiting value is returned in V
    """
    alpha0 = a1
    gamma0 = b1

    diff = 5
    n_its = 1

    while diff > 1e-15:

        alpha1 = alpha0.dot(alpha0)
        gamma1 = gamma0 + np.dot(alpha0.dot(gamma0), alpha0.T)

        diff = np.max(np.abs(gamma1 - gamma0))
        alpha0 = alpha1
        gamma0 = gamma1

        n_its += 1

        if n_its > max_it:
            raise ValueError('Exceeded maximum iterations of %i.' % (max_it) +
                             ' Check your input matrices')

    return gamma1


def doubleo(A, C, Q, R, tol=1e-15):
    """
    This function uses the "doubling algorithm" to solve the Riccati
    matrix difference equations associated with the Kalman filter.  The
    returns the gain K and the stationary covariance matrix of the
    one-step ahead errors in forecasting the state.

    The function creates the Kalman filter for the following system:

    .. math::

        x_{t+1} = A * x_t + e_{t+1}

        y_t = C * x_t + v_t

    where :math:`E e_{t+1} e_{t+1}' =  Q`, and :math:`E v_t v_t' = R`,
    and :math:`v_s' e_t = 0 \\forall s, t`.

    The function creates the observer system

    .. math::

        xx_{t+1} = A xx_t + K a_t

        y_t = C xx_t + a_t

    where K is the Kalman gain, :math:`S = E (x_t - xx_t)(x_t - xx_t)'`,
    and :math:`a_t = y_t - E[y_t| y_{t-1}, y_{t-2}, \dots ]`, and
    :math:`xx_t = E[x_t|y_{t-1},\dots]`.

    Parameters
    ----------
    A : array_like, dtype=float, shape=(n, n)
        The matrix A in the law of motion for x

    C : array_like, dtype=float, shape=(k, n)

    Q : array_like, dtype=float, shape=(n, n)

    R : array_like, dtype=float, shape=(k, k)

    tol : float, optional(default=1e-15)

    Returns
    -------
    K : array_like, dtype=float
        The Kalman gain K

    S : array_like, dtype=float
        The stationary covariance matrix of the one-step ahead errors
        in forecasting the state.

    Notes
    -----
    By using DUALITY, control problems can also be solved.
    """
    a0 = A.T
    b0 = np.dot(C.T, solve(R, C))
    g0 = Q
    dd = 1.
    ss = max(A.shape)
    v = np.eye(ss)

    # NOTE: This is a little hack to make sure we update k1 and k0 properly
    #       depending on the dimensions of C
    c_vec = C.shape[0] > 1

    while dd > tol:
        a1 = np.dot(a0, solve(v + np.dot(b0, g0), a0))
        b1 = b0 + np.dot(a0, solve(v + np.dot(b0, g0), np.dot(b0, a0.T)))
        g1 = g0 + np.dot(np.dot(a0.T, g0), solve(v + np.dot(b0, g0), a0))

        if c_vec:
            k1 = np.dot(A.dot(g1), solve(np.dot(C, g1.T).dot(C.T) + R.T, C).T)
            k0 = np.dot(A.dot(g0), solve(np.dot(C, g0.T).dot(C.T) + R.T, C).T)
        else:
            k1 = np.dot(np.dot(A, g1), C.T / (np.dot(C, g1).dot(C.T) + R))
            k0 = np.dot(A.dot(g0), C.T / (np.dot(C, g0).dot(C.T) + R))

        dd = np.max(np.abs(k1 - k0))
        a0 = a1
        b0 = b1
        g0 = g1

    return k1, g1


def doublex9(A, B, D, Q, R, sig):
    '''
    function[F,K,P,Pt]=doublex9(A,B,D,Q,R,sig)
    Revised Feb 15, 2001
    Solves the undiscounted robust control problem

        min sum (x'Qx + u'R u) for the state space system

        x' = A x + B u + D w
        sig < 0 indicates preference for robustness
        sig = -1/theta where theta is the robustness multiplier.

    Please note that because this is a MINIMUM problem, the convention
    is that Q and R are `positive definite' matrices (subject to the usual
    detectability qualifications).
    The optimal control with observed state is
        u_t = - F x_t

    The value function is -x'Px; note that the program returns a positive
    semi-definite P. Pt is the matrix D(P) where D(.) is the operator
    described in Hansen-Sargent.  The conservative measure
    of continuation value is -y' D(P) y = -y' Pt y,
    where y=x', next period's state.
    The program also returns a worst-case shock  matrix K
    where w_{t+1} = K x_t is the worst case shock.

    For problems with a target vector minimization
      min sum (z'z) with
       z  = H x + G u
    we have  R = G'G,   Q = H'H, and require H'G=0.
    Note that this is a minimum problem.
    If you want to combine this with robust estimation of
    the state, also use doublex8 to compute altered rule Fbar.
    doublex9 uses doubling algorithm from Anderson, et. al.
    Initializes with po=I.  The version is designed to find the optimal
    stable work solution in, for example,
    `permanent income' models that don't satisfy the detectability condition.
    See AHMS (Computational economics handbook chapter) for details.  doublex9
    adapts the AHMS algorithm to the risk-sensitive specicfication of
    Hansen-Sargent (IEEE, May, 1995)
    '''
    # Need to make sure R is an array
    if type(R) != np.ndarray:
        R = np.array([[R]])


    tol=1e-15
    dd=1

    ss = max(A.shape)
    v = np.eye(ss)
    po = v
    J = np.dot(B, solve(R, B.T)) + sig * np.dot(D, D.T)
    t = v + np.dot(J, po)

    a0 = solve(t, A)
    b0 = solve(t, J)
    g0 = Q - po + np.dot(A.T, po.dot(a0))  # note Q is H'*H

    while dd>tol:
        a1 = np.dot(a0, solve((v + b0.dot(g0)), a0))
        b1 = b0 + np.dot(a0, solve(v+b0.dot(g0), np.dot(b0, a0.T)))
        g1 = g0 + np.dot(a0.T, g0.dot(solve((v+b0.dot(g0)), a0)))
        dd = norm(g1 - g0) / norm(g0)
        a0 = a1
        b0 = b1
        g0 = g1

    gg1 = g1 + po   # gg1=g1+po  (changed from  paper)
    P = gg1
    # F=B'*P*((v+J*P)\A)    original formula from warehouse
    # g1t=P/(v+sig*P*D*D')  Change of sign to match the doublex4.m convention

    #  this line is taken from doublex4.m to take care of
    g1t = solve(v + np.dot(sig, gg1.dot(D.dot(D.T))), gg1)
    # transpose problem corrected april 14, 2000


    F = solve(np.dot(B.T, g1t.dot(B)) + R, np.dot(B.T, g1t.dot(A)))
    Pt = g1t
    rD, cD = D.shape
    K = -sig * (inv(np.eye(cD) + sig * np.dot(D.T, P.dot(D))).dot(
        np.dot(D.T, P.dot(A - B.dot(F)))))

    return F, K, P, Pt


def olrp(beta, A, B, Q, R, W=None, tol=1e-6, max_iter=1000):
    """
    Calculates F of the feedback law:

    .. math::

          U = -Fx

     that maximizes the function:

     .. math::

        \sum \{beta^t [x'Qx + u'Ru +2x'Wu] \}

     subject to

     .. math::
          x_{t+1} = A x_t + B u_t

    where x is the nx1 vector of states, u is the kx1 vector of controls

    Parameters
    ----------
    beta : float
        The discount factor from above. If there is no discounting, set
        this equal to 1.

    A : array_like, dtype=float, shape=(n, n)
        The matrix A in the law of motion for x

    B : array_like, dtype=float, shape=(n, k)
        The matrix B in the law of motion for x

    Q : array_like, dtype=float, shape=(n, n)
        The matrix Q from the objective function

    R : array_like, dtype=float, shape=(k, k)
        The matrix R from the objective function

    W : array_like, dtype=float, shape=(n, k), optional(default=0)
        The matrix W from the objective function. Represents the cross
        product terms.

    tol : float, optional(default=1e-6)
        Convergence tolerance for case when largest eigenvalue is below
        1e-5 in modulus

    max_iter : int, optional(default=1000)
        The maximum number of iterations the function will allow before
        stopping

    Returns
    -------
    F : array_like, dtype=float
        The feedback law from the equation above.

    P : array_like, dtype=float
        The steady-state solution to the associated discrete matrix
        Riccati equation

    """
    m = max(A.shape)
    rc, cb = np.atleast_2d(B).shape

    if W is None:
        W = np.zeros((m, cb))

    if type(R) != np.ndarray:
        R = np.array([[R]])

    if np.min(np.abs(eig(R)[0])) > 1e-5:
        A = sqrt(beta) * (A - B.dot(solve(R, W.T)))
        B = sqrt(beta) * B
        Q = Q - W.dot(solve(R, W.T))
        # k, s are different than in the matlab
        k, s = doubleo(A.T, B.T, Q, R)

        f = k.T + solve(R, W.T)

        p = s

    else:
        p0 = -0.1 * np.eye(m)
        dd = 1
        it = 1

        for it in range(max_iter):
            f0 = solve(R + beta * B.T.dot(p0).dot(B),
                       beta * B.T.dot(p0).dot(A) + W.T)
            p1 = beta * A.T.dot(p0).dot(A) + Q - \
                (beta * A.T.dot(p0).dot(B) + W).dot(f0)
            f1 = solve(R + beta * B.T.dot(p1).dot(B),
                       beta * B.T.dot(p1).dot(A) + W.T)
            dd = np.max(f1 - f0)
            p0 = p1

            if dd > tol:
                break
        else:
            msg = 'No convergence: Iteration limit of {0} reached in OLRP'
            raise ValueError(msg.format(max_iter))

        f = f1
        p = p1

    return f, p


def olrprobust(beta, A, B, C, Q, R, sig):
    """
    Solves the robust control problem

    :math:`min sum beta^t(x'Qx + u'R u)` for the state space system

    .. math::

        x' = A x + B u + C w

    olrprobust solves the problem by tricking it into a stacked olrp
    problem. as in Hansen-Sargent, Robustness in Macroeconomics,
    chapter 2. The optimal control with observed state is

    ..math::

        u_t = - F x_t

    And the value function is :math:`-x'Px`

    Parameters
    ==========
    beta : float
        The discount factor in the robust control problem.

    A, B, C : array_like, dtype = float
        The matrices A, B, and C from the state space system

    Q, R : array_like, dtype = float
        The matrices Q and R from the robust control problem

    sig :
        The robustness parameter. sig < 0 indicates a preference for
        robustness. sig = -1 / theta, where theta is the robustness
        multiplier.


    Returns
    =======
    F : array_like, dtype = float
        The optimal control matrix from above above

    P : array_like, dtype = float
        The psoitive semi-definite matrix defining the value function

    Pt : array_like, dtype = float
        The matrix D(P), wehre D(.) is the described in Hansen-Sargent.
        The conservative measure of continuation value is
        :math:`-y' D(P) y = -y' Pt y`, where :math:`y = x'` is next
        period's state.

    K : array_like, dtype = float
        the worst-case shock matrix K, where :math:`w_{t+1} = K x_t` is
        the worst case shock


    Notes
    =====
    Please note that because this is a MINIMUM problem, the convention
    is that Q and R are `positive definite' matrices (subject to the
    usual detectability qualifications).

    See Also
    ========
    doublex9

    """
    theta = -1/sig
    Ba = np.hstack([B, C])
    R, C, B = map(np.atleast_2d, [R, C, B])
    rR, cR = R.shape
    rC, cC = C.shape

    #  there is only one shock
    Ra = np.vstack([np.hstack([R, np.zeros((rR, cC))]),
                    np.hstack([np.zeros((cC, cR)), -beta*np.eye(cC)*theta])])

    f, P = olrp(beta, A, Ba, Q, Ra)
    rB, cB = B.shape
    F = f[:cB, :]
    rf, cf = f.shape
    K = -f[cB:rf, :]
    cTp = dot(C.T, P)  # Equivalent to Matlab C'*P
    Pt = P + theta**(-1)*dot(dot(P, C),
                             dot(inv(eye(cC) - theta**(-1)*dot(cTp, C)), cTp))

    return F, K, P, Pt


def Kworst(beta, sig, F, A, B, C, Q, R):
    '''
    For a risk-sensitivity parameter sig=-1/theta,
    this function computes the worst-case shock process Kw*x and value
    function matrix Pw for an arbitrary decision rule F for the state space
    system beta, A, B, C, Q, R.   It also returns the matrices BigO and littleo
    for forming  the associated discounted  entropy.
    '''

    if type(R) != np.ndarray:
        R = np.array([[R]])

    theta = -1./sig
    # costs of perturbation tricked into linear requlator
    Ra = -beta * theta
    # indirect quadratic form in state given F
    Qa = Q + np.dot(F.T, R.dot(F))
    # closed loop system matrix
    ABF = A - np.dot(B, F)
    # Kw, Pw were different in matlab
    Kw, Pw = olrp(beta, ABF, C, Qa, Ra)

    Kw = -Kw

    CChat = np.dot(C, inv(1. + sig*np.dot(C.T, Pw.dot(C))).dot(C.T))
    pw = (beta/(1.-beta)) * np.trace(Pw.dot(CChat))

    # Remark:

    # The worst case law of motion has transition matix ABF + C*K
    # Now compute matrices in quadratic form plus constant for discounted
    # entropy according to the formulas

    # O = beta*K'*K + beta*(ABF + CK)'*O*(ABF+CK)

    # o = h_o + .5*beta*trace(O*hat C * hat C') + beta* o

    # Tom: double check the formulas for .5's.

    ho = .5 * np.trace(inv(1. + sig * np.dot(C.T, Pw.dot(C))) - 1.) -\
         .5 * np.log(inv(1 + sig * np.dot(C.T, Pw.dot(C))))

    Sigworst = np.dot(C, inv(1. + sig * np.dot(C.T, Pw.dot(C)))).dot(C.T)
    AO = sqrt(beta) * (ABF + C.dot(Kw))
    BigO = doublej(AO.T, beta*Kw.T.dot(Kw))

    littleo = solve(np.array([[1. - beta]]), (ho + beta * np.trace(BigO.dot(Sigworst))))

    return Kw, Pw, pw, BigO, littleo
