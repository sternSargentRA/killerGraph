function[F,K,P,Pt]=doublex9(A,B,D,Q,R,sig)
%function[F,K,P,Pt]=doublex9(A,B,D,Q,R,sig)
% Revised Feb 15, 2001
% Solves the undiscounted robust control problem 
%
%  min sum (x'Qx + u'R u) for the state space system
%
%   x' = A x + B u + D w
%   sig < 0 indicates preference for robustness
%   sig = -1/theta where theta is the robustness multiplier.
% Please note that because this is a MINIMUM problem, the convention
% is that Q and R are `positive definite' matrices (subject to the usual
% detectability qualifications).  
% The optimal control with observed state is 
%  u_t = - F x_t
%  
%  The value function is -x'Px; note that the program returns a positive
%  semi-definite P. Pt is the matrix D(P) where D(.) is the operator
%  described in Hansen-Sargent.  The conservative measure
%  of continuation value is -y' D(P) y = -y' Pt y,
%  where y=x', next period's state.
%  The program also returns a worst-case shock  matrix K
%  where w_{t+1} = K x_t is the worst case shock.
%
%  For problems with a target vector minimization
%  min sum (z'z) with
%   z  = H x + G u
% we have  R = G'G,   Q = H'H, and require H'G=0.
% Note that this is a minimum problem.  
% If you want to combine this with robust estimation of
% the state, also use doublex8 to compute altered rule Fbar.  
% doublex9 uses doubling algorithm from Anderson, et. al.
% Initializes with po=I.  The version is designed to find the optimal
% stable work solution in, for example,  
% `permanent income' models that don't satisfy the detectability condition.
%  See AHMS (Computational economics handbook chapter) for details.  doublex9
%  adapts the AHMS algorithm to the risk-sensitive specicfication of
%  Hansen-Sargent (IEEE, May, 1995)

ss=max(size(A));
v=eye(ss);
po=v;
J=B*(R\B')+sig*D*D';
t=v+J*po;
a0=t\(A);
b0=t\J;
g0=Q-po+A'*po*a0;   % note Q is H'*H
tol=1e-15;
dd=1;
while dd>tol
a1=a0*((v+b0*g0)\a0);
b1=b0+a0*((v+b0*g0)\(b0*a0'));
g1=g0+a0'*g0*((v+b0*g0)\a0);
dd=norm(g1-g0)/norm(g0);
a0=a1;
b0=b1;
g0=g1;
end
gg1=g1+po;   %  gg1=g1+po % (changed from  paper)
P=gg1;
% F=B'*P*((v+J*P)\A);   % original formula from warehouse
%g1t=P/(v+sig*P*D*D');   %  Change of sign to match the doublex4.m convention

g1t=(v+sig*gg1*D*D')\gg1;  %  this line is taken from doublex4.m to take care of
                           % transpose problem; corrected april 14, 2000


F=(B'*g1t*B+R)\(B'*g1t*A);
Pt=g1t;
[rD,cD]=size(D);
K=-sig*(inv(eye(cD)+sig*D'*P*D))*D'*P*(A-B*F);
