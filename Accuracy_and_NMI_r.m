function [new_clusters,compare_indi] = Accuracy_and_NMI_r( Clus_reslut, ground_truth)

    [N,K] = size(Clus_reslut);
    new_clusters = zeros(N,K);
    Res_size = zeros(1,K);%each cluster result size
    Gro_size = zeros(1,K);%each cluster of truth size
    K_res = {};%each cluster of result mapping object index
    K_gro = {};%each cluster of ground truth mapping object index
    for k=1:K
        tt = [];
        K_res{1,k}=tt;
        K_gro{1,k}=tt;
    end
    % group the objects into each cluster
    for i = 1:N
        for k = 1:K
            if Clus_reslut(i,k)>0
               Res_size(1,k) =  Res_size(1,k)+1;
               K_res{1,k}(1,Res_size(1,k))=i;
            end
            if ground_truth(i,k)>0
               Gro_size(1,k)=Gro_size(1,k)+1;
               K_gro{1,k}(1,Gro_size(1,k))=i;
            end
        end
    end
    same_pencent = zeros(K,K);%the number of the same object pairs
    for groK = 1:K
        for resK = 1:K
            for i = 1:Gro_size(1,groK)
                ishas = findele(K_res{1,resK},K_gro{1,groK}(1,i));
                if ishas~=-1
                    same_pencent(groK,resK) = same_pencent(groK,resK)+1;
                end
            end
        end
        same_pencent(groK,:)=same_pencent(groK,:)/Gro_size(1,groK);
    end
    rank_same_pencent = zeros(K,K);
    for i=1:K
        [temp,index] = sort(same_pencent(i,:),'descend');
        for j = 1:K
            rank_same_pencent(i,index(j))=j;
        end
    end
    compare_indi = zeros(1,K);%result map to ground_truth
    adjust = 1;
    while(adjust)
        adjust =0;
        for j = 1:K
            hasone = find(rank_same_pencent(:,j)==1);
            if length(hasone) ~= 1
               adjust = 1;
               break;
            end
        end
        if adjust == 1
            for i = 1:K
                for j = 1:K
                    if rank_same_pencent(i,j)==1;
                        hasone = find(rank_same_pencent(:,j)==1);
                        if length(hasone)~=1
                            for tt = 1:length(hasone) 
                                if hasone(tt) ~= i
                                    if same_pencent(i,j)<same_pencent(hasone(tt),j)
                                        rank_same_pencent(i,:)=rank_same_pencent(i,:)-1;
                                        break;
                                    end
                                    if same_pencent(i,j)>same_pencent(hasone(tt),j)
                                        rank_same_pencent(hasone(tt),:)=rank_same_pencent(hasone(tt),:)-1;
                                        break;
                                    end
                                    if same_pencent(i,j)==same_pencent(hasone(tt),j)
                                        pos1 = find(rank_same_pencent(i,:)==2);
                                        pos1 = find(rank_same_pencent(hasone(tt),:)==2);
                                        if same_pencent(i,pos1)>same_pencent(hasone(tt),pos1)
                                            rank_same_pencent(i,:)=rank_same_pencent(i,:)-1;
                                        else
                                            rank_same_pencent(hasone(tt),:)=rank_same_pencent(hasone(tt),:)-1;
                                        end
                                        break;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
               
    end
    for i = 1:K
        for j = 1:K
            if rank_same_pencent(i,j)~=1
                rank_same_pencent(i,j)=0;
            else
                compare_indi(1,i)=j;
            end
        end
    end
    for i = 1:K
        new_clusters(:,i)=Clus_reslut(:,compare_indi(1,i));
    end
    for i = 1:N
        for j = 1:K
            if ground_truth(i,j)==1
                if new_clusters(i,j)>0
                    new_clusters(i,j)=1;
                end
            else
                new_clusters(i,j)=0;
            end
        end
    end
end




