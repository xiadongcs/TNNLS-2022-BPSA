clc
close all

%% 
addpath('data'); addpath('functions'); 
Files = dir(fullfile('data', '*.mat'));
Max_datanum = length(Files);

%% 
for data_num = 1:Max_datanum  
    Dname = Files(data_num).name;
    disp(['***********The test data name is: ***' num2str(data_num) '***'  Dname '****************'])
    load(Dname);
    
   %% 
    file_path = 'Results/';
    folder_name = Dname(1:end-4); 
    file_path_name = strcat(file_path,folder_name);
    if exist(file_path_name,'dir') == 0   
       mkdir(file_path_name);
    end
    file_mat_path = [file_path_name '/'];
    
    classnum = length(unique(Y));
    knn = 10; fla1 = 1:1:4; fla2 = 1:1:4; 
    for j = 1:length(fla1)
        flag1 = fla1(j);
        for k = 1:length(fla2)
            flag2 = fla2(k);
            
             [lab,H,obj_o,alpha,beta] = BPSA(X,classnum,knn,flag1,flag2);
             result_BPSA = ClusteringMeasure(Y,lab);
            
             file_name = [folder_name '_Uv_' num2str(flag1) '_H_' num2str(flag2) '.mat'];
             save ([file_mat_path,file_name],'Dname','result_BPSA','lab','H','obj_o','alpha','beta');
        end
    end
end