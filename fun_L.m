function [ err ] = fun_L( input_X,U,T,lambda )
%FUN_L Summary of this function goes here
%   Detailed explanation goes here
% f=norm(input_X-(ktensor(U)))^2;
f=norm(input_X)^2+norm((ktensor(U)))^2-2*innerprod(input_X,ktensor(U));
g=0;
for t=1:T
    g = g+norm(U{1,t})^2;
end
err = 0.5*f-0.5*lambda*g;
end

