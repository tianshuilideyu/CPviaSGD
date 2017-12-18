
clear;
clc;


bb = 1;
error = cell(1,10);
res = cell(1,10);
res_tem = cell(1,10);
AC_rec = cell(1,10);
NMI_rec = cell(1,10);
AC_avg = cell(1,10);
NMI_avg = cell(1,10);
total_time = cell(1,10);
iter=cell(1,10);
lambda = 0.001;
eta = 1;
choice = 2;%ALS(0) OR SGD(1) OR SOS(2)
while 1
    if bb > 10
        break;
    end
    random_inital;
    load('initial_U.mat');
    iterNum = 0;
    thr = 0.000001; %threshold vale;
    fprintf('SGD is start \n');                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    err=zeros(1,1000); %error
    f_curr=0;
    f_pre=1;    
    tic
     while 1
        iterNum = iterNum+1;
        eta = 1/(iterNum+1);
        for t=1:T
            U{1,t}=UpdateU(U,t,input_X,T,K,eta,lambda,choice);
%             U{1,t}=Row_Normalize(U{1,t},N(t),K);
        end
        f_curr=fun_L( input_X,U,T,lambda );
        err(1,iterNum)=abs((f_curr-f_pre)/f_pre);
        f_pre=f_curr;
        formatSpec = 'This is %2.0f times iteration. The error is %8.7f.\n';
        fprintf(formatSpec,iterNum,err(1,iterNum));
        if iterNum>=1000 ||err(1,iterNum)<thr
            break;
        end
     end
    spend_time = toc;
    fprintf('SGD is done \n');
    error{1,bb}=err;
    total_time{1,bb}=spend_time;
    iter{1,bb}=iterNum;
    
    for t=1:T
       U{1,t}=Row_Normalize(U{1,t},N(t),K);
    end
    res{1,bb}=U;
%     
    if realworldornot == 0
        U_tem = U;
        for t=1:T
            [r,c]=size(U_tem{1,t});
            for i=1:r
                temp_max = 0;
                temp_ind = 0;
                for j=1:c
                    if U_tem{1,t}(i,j)>temp_max
                        temp_max = U_tem{1,t}(i,j);
                        temp_ind = j;
                    end
                end
                U_tem{1,t}(i,:)=0;
                U_tem{1,t}(i,temp_ind) = 1;
            end
        end
        ac = zeros(1,T);
        rand_index = zeros(1,T);
        match_index = {};
        NMI = zeros(1,T); 
        for t = 1:T
%             [K_clus_res,clus_res_size] = ClusterResultOperator(U_tem{1,t});
%             [K_grou_tru,grou_tru_size] = ClusterResultOperator(U_groundtruth{1,t});
%             [ac(1,t),rand_index(1,t),match_index{1,t}]=AccMeasure(K_grou_tru,K_clus_res);
%             NMI(1,t) = Normalized_mutual_information(K_clus_res,K_grou_tru);
            [clus{1,t},comid{1,t}]=Accuracy_and_NMI_r( U_tem{1,t},U_groundtruth{1,t});
            [K_clus_res,clus_res_size] = ClusterResultOperator_r(clus{1,t});
            [K_grou_tru,grou_tru_size] = ClusterResultOperator_r(U_groundtruth{1,t});
            [ac(1,t),rand_index(1,t),match{1,t}]=AccMeasure(K_grou_tru,K_clus_res);
            NMI(1,t) = Normalized_mutual_information(K_clus_res,K_grou_tru);
        end
        aver_ac=sum(ac.*N)/sum(N);
        aver_nmi=sum(NMI.*N)/sum(N);
        AC_avg{1,bb}=aver_ac;
        NMI_avg{1,bb}=aver_nmi;
    else
        U_tem = U;
        for t=1:T
            [r,c]=size(U_tem{1,t});
            for i=1:r
                for j=1:c
                    if U_tem{1,t}(i,j)<0.01
                        U_tem{1,t}(i,j)=0;
                    end
                end
            end
            U_tem{1,t} = Row_Normalize(U_tem{1,t},r,c);
        end
        ac = zeros(1,2);
        rand_index = zeros(1,2);
        match={};
        NMI = zeros(1,2); 
        for t = 1:2
            [clus{1,t},comid{1,t}]=Accuracy_and_NMI_r( U_tem{1,t},U_groundtruth{1,t});
            [K_clus_res,clus_res_size] = ClusterResultOperator_r(clus{1,t});
            [K_grou_tru,grou_tru_size] = ClusterResultOperator_r(U_groundtruth{1,t});
            [ac(1,t),rand_index(1,t),match{1,t}]=AccMeasure(K_grou_tru,K_clus_res);
            NMI(1,t) = Normalized_mutual_information(K_clus_res,K_grou_tru);
        end
          
        aver_ac=sum(ac.*N(1:2))/sum(N(1:2));
        aver_nmi=sum(NMI.*N(1:2))/sum(N(1:2));
        AC_rec{1,bb} = ac;
        AC_avg{1,bb}=aver_ac;
        NMI_rec{1,bb} = NMI;
        NMI_avg{1,bb}=aver_nmi;
    end
    
    res_tem{1,bb}=U_tem;
% 
%    
    bb = bb+1;
%     
    clear iterNum;
    clear err;
    clear U;
    clear U_tem;
    %clear ac;
    %clear NMI;
    clear rand_index;
    clear match_index;
    clear aver_ac;
    clear aver_nmi;
    clear T;
    clear K;
    clear U_groundtruth;
    clear N; 
    clear M;
    clear input_X;
    clear spend_time;
%     
end
%save results
save('STFClus_result.mat','choice','error','res','res_tem','AC_rec','AC_avg','NMI_rec','NMI_avg','total_time','iter','lambda','eta');
