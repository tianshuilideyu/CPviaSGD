function [K_res,Clus_size] = ClusterResultOperator_r (clusterResult)
	
    [N,K] = size(clusterResult);
    Clus_size = zeros(1,K);%each cluster size
    K_res = zeros(1,N);%the cluster label of each object
    
    for i = 1:N
        for k=1:K
            if clusterResult(i,k) > 0
                Clus_size(1,k)=Clus_size(1,k)+1;
                K_res(1,i)=k;
            end
        end
    end   
    
end
