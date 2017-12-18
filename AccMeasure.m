function [Acc,rand_index,match_index]=AccMeasure(T,idx)
%Measure percentage of Accuracy and the Rand index of clustering results
% The number of class must equal to the number cluster 

%Output
% Acc = Accuracy of clustering results
% rand_index = Rand's Index,  measure an agreement of the clustering results
% match_index = 2xk mxtrix which are the best match of the Target and clustering results

%Input
% T = 1xn ground truth
% idx =1xn matrix of the clustering results

k=max(T);
n=length(T);
for i=1:k
    temp=find(T==i);
    a{i}=temp; %#ok<AGROW>
end

b1=[];
t1=zeros(1,k);
for i=1:k
    tt1=find(idx==i);
    for j=1:k
       t1(j)=sum(ismember(tt1,a{j}));
    end
    b1=[b1;t1]; %#ok<AGROW>
end
    Members=zeros(1,k); 
    
P = perms((1:k));
    Acc1=0;
for pi=1:size(P,1)
    for ki=1:k
        Members(ki)=b1(P(pi,ki),ki);
    end
    if sum(Members)>Acc1
        match_index=P(pi,:);
        Acc1=sum(Members);
    end
end

rand_ss1=0;
rand_dd1=0;
for xi=1:n-1
    for xj=xi+1:n
        rand_ss1=rand_ss1+((idx(xi)==idx(xj))&&(T(xi)==T(xj)));
        rand_dd1=rand_dd1+((idx(xi)~=idx(xj))&&(T(xi)~=T(xj)));
    end
end
rand_index=200*(rand_ss1+rand_dd1)/(n*(n-1));
Acc=Acc1/n; 
match_index=[1:k;match_index];
