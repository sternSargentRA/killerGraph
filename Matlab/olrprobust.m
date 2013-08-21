function[F,K,P,Pt]=olrprobust(beta,A,B,C,Q,R,sig)
%function[F,K,P,Pt]=olrprobust(beta,A,B,C,Q,R,sig)
% Solves the robust control problem 
%
%  min sum beta^t(x'Qx + u'R u) for the state space system
%
%   x' = A x + B u + C w
%   sig < 0 indicates preference for robustness.
%   sig = -1/theta where theta is the robustness multiplier.
% Please note that because this is a MINIMUM problem, the convention
% is that Q and R are `positive definite' matrices (subject to the usual
% detectability qualifications).  
% 
% olrprobust solves the problem by tricking it into a stacked olrp problem.  
% as in Hansen-Sargent, Robustness in Macroeconomics, chapter 2.
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
%  See also: doublex9.m  
theta=-1/sig;
Ba=[B C];
[rR,cR]=size(R);
[rC,cC]=size(C);
Ra=[R zeros(rR,cC);
    zeros(cC,cR) -beta*eye(cC)*theta];      %  there is only one shock

[f,P] = olrp(beta,A,Ba,Q,Ra);
[rB,cB]=size(B);ra=length(A);
F=f(1:cB,:);[rf,cf]=size(f);
K=-f(cB+1:rf,:);
Pt=P+theta^(-1)*P*C*inv(eye(size(C'*P*C))-theta^(-1)*C'*P*C)*C'*P;

