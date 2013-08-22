import numpy as np
from numpy import dot, eye
from killergraphfuncs import olrp
from scipy.linalg import inv


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
