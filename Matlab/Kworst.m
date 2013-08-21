function [Kw,Pw,pw,BigO,littleo] = Kworst(beta,sig,F,A,B,C,Q,R)
%function [Kw,Pw,pw,BigO,littleo] = Kworst(beta,sig,F,A,B,C,Q,R)
% For a risk-sensitivity parameter sig=-1/theta, 
% this function computes the worst-case shock process Kw*x and value
% function matrix Pw for an arbitrary decision rule F for the state space
% system beta, A, B, C, Q, R.   It also returns the matrices BigO and littleo
% for forming  the associated discounted  entropy. 
theta = -1/sig;
Ra=  -beta*theta;  % costs of perturbation tricked into linear requlator
Qa=Q+F'*R*F;  %  indirect quadratic form in state given F
ABF=A-B*F;    % closed loop system matrix
[Kw,Pw]=olrp(beta,ABF,C,Qa,Ra)

Kw=-Kw;

CChat=C*(inv(1+sig*C'*Pw*C))*C';
pw=(beta/(1-beta))*trace(Pw*CChat);

% remark:
% 
%  The worst case law of motion has transition matix ABF + C*K


%  Now compute matrices in quadratic form plus constant for discounted
%  entropy according to the formulas
%  
% O = beta*K'*K + beta*(ABF + CK)'*O*(ABF+CK)
%
% o = h_o + .5*beta*trace(O*hat C * hat C') + beta* o
%
%  Tom: double check the formulas for .5's.   
%

ho=.5*trace(inv(1+sig*C'*Pw*C)-1) -.5*log(inv(1+sig*C'*Pw*C))
Sigworst=C*inv(1+sig*C'*Pw*C)*C';
AO=sqrt(beta)*(ABF+C*Kw);
BigO=doublej(AO',beta*Kw'*Kw);  % note the ' on AO
%   BigO=dlyap(beta*Kw'*Kw,AO');  % note where I have put the ' 
                                  % dlyap doesn't seem to work
littleo=(1-beta)\(ho+beta*trace(BigO*Sigworst))




end

