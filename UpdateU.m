function [ Ut ] = UpdateU( U,t,input_X,T,K,eta,lambda,choice)
%UPDATEU Summary of this function goes here
%   Detailed explanation goes here

Gamma=ones(K,K);
for i = 1:T
    if i~=t
        Gamma = Gamma.*(U{1,i}'*U{1,i});
    end
end
I = eye(K,K);
if choice == 0
    Ut=mttkrp(input_X,U,t)*inv((Gamma+lambda*I)); %%ALS
end
if choice == 1
    Ut=U{1,t}*(I-eta*(Gamma+lambda*I))+eta*(-mttkrp(input_X,U,t)); %%SGD
end
if choice == 2
	Ut=eta*mttkrp(input_X,U,t)*inv((Gamma+lambda*I))+(1-eta)*U{1,t};  %%SOS
end

