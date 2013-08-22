% monopolist_robust_mock3.m
%  This script file modifies *_mock2.m to put in a loop that generates
%  the "killer graph"
%
% monopolist with adjustment costs
% created as an example in terms of which to cast the `killer graph'
% showing robustness. We'll use olrprobust.m to compute key objects
%
% Inverse demand curve:
%
%  p_t = A_0 - A_1 Q_t + d_t
%
%  d_{t+1} = \rho d_t + \sigma_d epsilon_{t+1}
%
%  epsilon_{t+1} is iid N(0,1)
%
% Period return function for monopolist:
% 
%  R_t =  p_t Q_t - .5 e (Q_{t+1}-Q_t) - c Q_t   =
%
%   (A_0 - c) Q_t - A_1 Q_t^2 _ d_t Q_t - .5 e Q_t 
%
% Objective of the firm:
%  
%  E_t \sum_{t=0}^\infty \beta^t R_t   
%
%
% Create linear regulator. State is x_t' = [1, Q_t, d_t]
% control is u_t = (Q_{t+1} - Q_t )
%
% The implied state-space matrices are
%
% the A, B, C, Q, R below.
%
% The free parameters are beta, A_0, A_1, rho, sigma_d, beta, e, c
% AND the robustness parameter sig \equiv - theta^{-1}
%


A_0=100; A_1=.5;
rho=.9; sigma_d=.05;
beta=.95;
c=2; e=50;

sig=-10;  % this is the risk sensitivity parameter and should be negative
           % when there is fear of model misspecification

Q = [0 .5*(A_0-c) 0; .5*(A_0-c) -A_1 .5;  0 .5 0];
R=-.5*e;

Q=-Q;
R=-R;     % flip signs to make both matrices `positive definite' because we'll solve a minimum problem.

A=[1 0 0; 0 1 0; 0 0 rho];
B=[0 1 0]';
C=[0 0 sigma_d]';

%  [F,K,P,Pt]=olrprobust(beta,A,B,C,Q,R,sig)  % I have commented out this
%  equivalent way of computing the decision rule.  I'll use doublex9.m
%  instead.  


% Use doublex9 to compute the robust deicsionrule and worst-case shock
% vector w = K x
%
% first adjust for discounting -- see Robustness, p. 52
%
% Ad=sqrt(beta)*A;
% Bd=sqrt(beta)*B;
% [F9,K9,P9,Pt9]=doublex9(Ad,Bd,C,Q,R,sig)
% 
% 
% K9=K9/sqrt(beta);

%  F9 is the robust decision rule affiliated with sig = - 1/theta;
%  K9 is the worst case shock coefficient matrix.




      




% Evaluate various objects to create the graph
%
% Put things together

[fo,po] = olrp(beta,A,B,Q,R)   % compute the optimal rule

% now compute a robust rule affiliated with some sig <0

sig=-50;
Ad=sqrt(beta)*A;
Bd=sqrt(beta)*B;
[F9,K9,P9,Pt9]=doublex9(Ad,Bd,C,Q,R,sig)


K9=K9/sqrt(beta);

%  F9 is the robust decision rule affiliated with sig = - 1/theta;
%  K9 is the worst case shock coefficient matrix affiliated with that rule and that sig.

% compute nonstochastic steady state
%
%  xs is the steady state for [1, Q, d] under the robust decision rule F
%  under the APPROXIMATING model
%

ABF=A-B*F9;
nsize=max(size(ABF)); 
e1=eye(nsize);
zz=null(e1-ABF);
xs=zz./zz(1);


% % check the positive definiteness of the worst-case covariance matrix to
% % assure that theta exceeds the breakdown point
% 
check = eye(size(P9))+sig*C'*P9*C

checkfinal=eig(check)





% Now compute two worst case shocks and associated value functions
% and entropies affiliated with some other sig called sigc

N=100;

Xopt=zeros(N,2);
Xrobust=zeros(N,2);

sigspace=linspace(.0000001,100,N);

for i=1:N

sigc=-sigspace(i);
%sigc=-10;
%sigc=-15;

[Kwo,Pwo,pwo,BigOo,littleoo] = Kworst(beta,sigc,fo,A,B,C,Q,R);

[Kwr,Pwr,pwr,BigOr,littleor] = Kworst(beta,sigc,F9,A,B,C,Q,R);


% Now compute value functions and entropies evaluated at the 
% initial state xo=[1 0 0 ]';

xo=[1 0 0 ]';

Vo= - xo'*Pwo*xo - pwo;
Vr= - xo'*Pwr*xo - pwr;

ento=xo'*BigOo*xo + littleoo;
entr=xo'*BigOr*xo + littleor;

Xopt(i,1)=Vo;
Xopt(i,2)=ento;

Xrobust(i,1) =Vr;
Xrobust(i,2)=entr;

end

clf

figure(1)
plot(Xopt(:,2),Xopt(:,1),'r')
hold on
plot(Xrobust(:,2),Xrobust(:,1),'b--')




% Now do the "optimistic" shock calculations

Yopt=zeros(N,2);
Yrobust=zeros(N,2);



for i=1:N

sigc=.1*sigspace(i);  % scale the space to fine tune the graph


[Kwo,Pwo,pwo,BigOo,littleoo] = Kworst(beta,sigc,fo,A,B,C,Q,R);

[Kwr,Pwr,pwr,BigOr,littleor] = Kworst(beta,sigc,F9,A,B,C,Q,R);


% Now compute value functions and entropies evaluated at the 
% initial state xo=[1 0 0 ]';

xo=[1 0 0 ]';

Vo= - xo'*Pwo*xo - pwo;
Vr= - xo'*Pwr*xo - pwr;

ento=xo'*BigOo*xo + littleoo;
entr=xo'*BigOr*xo + littleor;

Yopt(i,1)=Vo;
Yopt(i,2)=ento;

Yrobust(i,1) =Vr;
Yrobust(i,2)=entr;

end

hold on
plot(Yopt(:,2),Yopt(:,1),'r')
hold on
plot(Yrobust(:,2),Yrobust(:,1),'b--')
xlabel('entropy')
ylabel('value function')
title('value sets')

print -dpdf killer_graph


