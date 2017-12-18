function [NMI] = Normalized_mutual_information(Clus_reslut,ground_truth)
% Nomalized mutual information

if length( Clus_reslut ) ~= length( ground_truth)
    error('length( Clus_reslut ) must == length( ground_truth)');
end
total = length(Clus_reslut);
A_ids = unique(Clus_reslut);
A_class = length(A_ids);
B_ids = unique(ground_truth);
B_class = length(B_ids);
% Mutual information
idAOccur = double (repmat( Clus_reslut, A_class, 1) == repmat( A_ids', 1, total ));
idBOccur = double (repmat( ground_truth, B_class, 1) == repmat( B_ids', 1, total ));
idABOccur = idAOccur * idBOccur';
Px = sum(idAOccur') / total;
Py = sum(idBOccur') / total;
Pxy = idABOccur / total;
MImatrix = Pxy .* log2(Pxy ./(Px' * Py)+eps);
MI = sum(MImatrix(:));
% Entropies
Hx = -sum(Px .* log2(Px + eps),2);
Hy = -sum(Py .* log2(Py + eps),2);
%Normalized Mutual information
% NMI = 2 * MI / (Hx+Hy);

NMI = MI / sqrt(Hx*Hy);% another version of NMI

end
