function [ Ut ] = Row_Normalize( Ut,Nt,K )
%ROW_NORMALIZE Summary of this function goes here
%   Detailed explanation goes here
% normalize the row of the input factor matrix
% Nt is the row number
Ut(Ut<0)=0;
for i = 1:Nt
    sumofrow = sum(Ut(i,:));
    if sumofrow == 0
        Ut(i,:)=rand(1,K);
        sumofrow = sum(Ut(i,:));
    end
    Ut(i,:) = Ut(i,:)/sumofrow;
end


end

