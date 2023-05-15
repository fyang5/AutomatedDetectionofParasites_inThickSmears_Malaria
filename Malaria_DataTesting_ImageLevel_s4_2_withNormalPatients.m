%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  This program test the CNN network trained by the SET1(120 patients) on
%  SET2 (30 infected patients + 10 normal patients) at "Image Level". 
%  Note: Set1 is generated from annotation files (postive) and 
%  greedy(negative). Set2 (both postive and negative) is generated 
%  directly by greedy method.
%
%  Feng Yang,
%  NLM/NIH
%  Bethesda, March 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;
imx = 44;
imy = 44;
%%load the CNN model
t0=tic;
net = load('net_train_new_fix1_withNormal_drop_2_05_val.mat');
% net = load('net_train_new_fix1_withNormal_drop_2_05_val_DataAug.mat');
trained_net  = net.net_new;
filename = 'ParasitePrediction_Test_new_fix1_withNormal_thresh_06.txt';
path_anno = 'C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\Bangladesh_may_2017\GT_updated\';
top_folder = 'C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\ProgforBangladeshData\Pats_Bangladesh_Jan2018\test_new_fix1_withNormal\GreedyPreselection\';
rad = imx*0.5;
%% testing on data which are generated using preselection
temp_folder = dir(top_folder);
temp_folder(1:2) = [];
num_folder = length(temp_folder);
file2 = fopen(filename,'w+');
fprintf(file2,'%s %s %s %s %s\n','foldername', 'filename_anno','num_para','Testing-1-folder','cnn_thresh' );
filename0 = 'ParaPred_Test_new_fix1_withNormal_thresh_06_infectedpart.txt';
file0 = fopen(filename0,'w+');
fprintf(file0,'%s %s %s %s %s\n','foldername', 'filename_anno','num_para','Testing-1-folder','cnn_thresh' );
filename1 = 'ParaPred_Test_new_fix1_withNormal_thresh_06_normalpart.txt';
file1 = fopen(filename1,'w+');
fprintf(file1,'%s %s %s %s %s\n','foldername', 'filename_anno','num_para','Testing-1-folder','cnn_thresh' );
% num_im = 0;
for i = 1:num_folder 
    foldername = temp_folder(i).name;
    pathname_anno = [top_folder, temp_folder(i).name,'\'];
    temp_anno = dir(pathname_anno);
    temp_anno(1:2) = [];
    num_anno = length(temp_anno);
    for j = 1:num_anno
%         num_im = num_im + 1;
        path_para = [pathname_anno,temp_anno(j).name,'\1\'];
        path_neg = [pathname_anno,temp_anno(j).name,'\0\'];
        file_anno = [path_anno,temp_folder(i).name,'\',temp_anno(j).name,'.txt'];
        s = dir(path_para);
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CASE2 :prediction on one image level
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [GT,lbls,mask,pos,num_para,Nlab] = loadGT_new_1(file_anno,imx);
        if num_para ~= 0
            if length(s)~=2
                loc_te_pos = path_para;
            else
                loc_te_pos = path_para;
                disp(['There is no positive patches in file:',file_anno]);
            end
            if length(dir(path_neg))~=2
                loc_te_neg = path_neg;
            else
                loc_te_neg = path_neg;
                disp(['There is no negative patches in file:',file_anno]);
            end
            ParaData_te = datastore(loc_te_pos,...
                'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
            ParaData_te.Labels = categorical(ones(size(ParaData_te.Files)));
            NegData_te = datastore(loc_te_neg,...
                'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
            NegData_te.Labels = categorical(zeros(size(NegData_te.Files)));
            testPatsData = datastore({loc_te_pos;loc_te_neg},'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
            testPatsData.Labels = [ParaData_te.Labels; NegData_te.Labels];
            %%read the labels of testing patches
            labelCount_te = countEachLabel(testPatsData);
            [predictedLabels,scores] = classify(trained_net,testPatsData,'ExecutionEnvironment','gpu');
            %%threshold = 0.6
            predictedLabels(find(scores(:,1)<0.6)) = categorical(0);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%added by yf 11/13/2018: histogram of the normal patients
%             valLabels = testPatsData.Labels;
%             ind=find(valLabels=='0');
%             for ii = 1:length(ind)
%                 score1(ii,1)=scores(ind(ii));
%             end
%             hist(score1)
%             %%added by yf 11/13/2018:histogram of the infected patients:
%             ind1=find(valLabels=='1');
%             for jj = 1:length(ind1)
%                 score2(jj,1)=scores(ind(jj));
%             end
%             hist(score2)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            num_cnn_06 = length(find(predictedLabels=='1'));
            fprintf(file0,'%s %s %d %d %d\n',foldername, temp_anno(j).name,num_para,length(s)-2, num_cnn_06);
        else
            if length(dir(path_neg))~=2
                loc_te_neg = path_neg;
            else
                loc_te_neg = path_neg;
                disp(['There is no negative patches in file:',file_anno]);
            end
            testPatsData = datastore(loc_te_neg,...
                'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
            testPatsData.Labels = categorical(zeros(size(testPatsData.Files)));
            %%read the labels of testing patches
            labelCount_te = countEachLabel(testPatsData);
            [predictedLabels,scores] = classify(trained_net,testPatsData,'ExecutionEnvironment','gpu');
            %%threshold = 0.6
            predictedLabels1 = predictedLabels;
            predictedLabels(find(scores(:,1)<0.6)) = categorical(0);
            num_cnn_06 = length(find(predictedLabels=='1'));
            fprintf(file1,'%s %s %d %d %d\n',foldername, temp_anno(j).name,num_para,length(s)-2, num_cnn_06);
        end
        fprintf(file2,'%s %s %d %d %d\n',foldername, temp_anno(j).name,num_para,length(s)-2, num_cnn_06);
    end
end
fclose(file2);
fclose(file0);
fclose(file1);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image Level and Patient Level estimation of the prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\
%%% to read the annotated wbc numbers and estimated wbc numbers
filename5 = 'WBCandParasite_PatientLevel_withNormal_withTPFP.txt';
fid5 = fopen(filename5,'r');
a = textscan(fid5,'%s %d %d %d %d %d %d %d %d %12.8f','headerlines',1);
fclose(fid5);
wbc_g = a{1,2}; wbc_g=wbc_g';
wbc = a{1,4};  wbc=wbc';

% filename = 'ParasitePrediction_Test_new_fix1_withNormal_thresh_07.txt';
fid = fopen(filename,'r');
a = textscan(fid,'%s %s %d %d %d %d\n','headerlines',1);
fclose(fid);
name_patient = a{1,1};
name_file = a{1,2};
Num_presel = double(a{1,4});
Num_ground = double(a{1,3});
num_cnn_06 = double(a{1,5});
%%correlation coefficent 
num_cnn = num_cnn_06;
R = corrcoef(num_cnn,Num_ground)
figure;plotregression(Num_ground,num_cnn);grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%correlation coefficent in patient level
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_pati = 1;
j=1;
while j<=length(name_patient)
    name_temp = name_patient{j};
    ind = find( strcmp(name_patient{j},name_patient)==1 ) ;
    num = length(ind);
    cnnsum_1 = sum(num_cnn_06(ind(1):ind(num)));
    groudsum = sum(Num_ground(ind(1):ind(num)));
    n_cnn_06(num_pati) = cnnsum_1;
    n_ground(num_pati) = groudsum;
    folder_pati{num_pati,1} = name_temp;
    patn = str2num(name_temp(3:5));
    if patn >199 & patn <251  %% the normal patient indices are from 200 to 250
        label_g(num_pati) = 0; % '0' indicates normal patients
    else
        label_g(num_pati) = 1;  % '1' indicates infected patients
    end
    j = j + num;
    num_pati = num_pati + 1;
end
R1 = corrcoef(n_cnn_06,n_ground)
figure;plotregression(n_ground,n_cnn_06);grid on;
parasitemia_ul = int32( 40*200*n_cnn_06./double(wbc) )
parasitemia_g = int32( 40*200*n_ground./double(wbc_g) )
ind_g = find(parasitemia_g==0)
ind = find(parasitemia_ul<400) %% 400 is an empirical value
label_pred = ones(size(label_g,1),size(label_g,2));
label_pred(parasitemia_ul<400)=0;
%%calculate the accuracy precision, recall and specificity
tp = length(find(label_pred==1& label_g==1));  %% tpn means true positive number
fp =  length(find(label_pred==1& label_g==0));
tn = length(find(label_pred==0& label_g==0));
fn = length(find(label_pred==0& label_g==1));
accur = (tp+tn)/(tp+fp+tn+fn)
preci = tp/(tp+fp)
recal = tp/(tp+fn)
speci = tn/(fp+tn)
f_score = 2*recal*preci/(recal+preci)
mcc = (tp*tn-fp*fn)/sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) )
%%ROC curve and AUC
[X,Y,T,AUC] = perfcurve(label_g,label_pred,1) ;
figure;plot(X,Y)
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for malaria diagnosis - patient level');grid on;
AUC
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image Level and Patient Level estimation of the prediction- separate the
% infected patients and normal patients: part1:Normal patients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen(filename1,'r');
a = textscan(fid,'%s %s %d %d %d %d\n','headerlines',1);
fclose(fid);
name_patient1 = a{1,1};
name_file1 = a{1,2};
Num_presel1 = double(a{1,4});
Num_ground1 = double(a{1,3});
num_cnn_1 = double(a{1,5});
%%for normal patients
% ind = find(Num_ground==0);
% for i = length(ind)
%     num_cnn_n(i) = num_cnn_06(ind(i));
%     num_ground_n(i) = 0;
% end
%%correlation coefficent 
R = corrcoef(num_cnn_1,Num_ground1)
figure;plotregression(Num_ground1,num_cnn_1);grid on;
mean(num_cnn_1)
%% correlation coefficent in patient level
num_pati1 = 1;
j=1;
while j<=length(name_patient1)
    name_temp = name_patient1{j};
    ind = find( strcmp(name_patient1{j},name_patient1)==1 ) ;
    num = length(ind);
    cnnsum_1 = sum(num_cnn_1(ind(1):ind(num)));
    groudsum = sum(Num_ground1(ind(1):ind(num)));
    n_cnn_1(num_pati1) = cnnsum_1;
    n_ground1(num_pati1) = groudsum;
    j = j + num;
    num_pati1 = num_pati1 + 1;
end
R1 = corrcoef(n_cnn_1,n_ground1)
figure;plotregression(n_ground1,n_cnn_1);grid on;
mean(n_cnn_1)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% part2:infected patients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen(filename0,'r');
a = textscan(fid,'%s %s %d %d %d %d\n','headerlines',1);
fclose(fid);
name_patient0 = a{1,1};
name_file0 = a{1,2};
Num_presel0 = double(a{1,4});
Num_ground0 = double(a{1,3});
num_cnn_0 = double(a{1,5});
%%for normal patients
% ind = find(Num_ground==0);
% for i = 1:length(ind)
%     num_cnn_n(i) = num_cnn_06(ind(i));
%     num_ground_n(i) = 0;
% end
%%correlation coefficent 
R = corrcoef(num_cnn_0,Num_ground0)
figure;plotregression(Num_ground0,num_cnn_0);grid on;
%% correlation coefficent in patient level
num_pati0 = 1;
j=1;
while j<=length(name_patient0)
    name_temp = name_patient0{j};
    ind = find( strcmp(name_patient0{j},name_patient0)==1 ) ;
    num = length(ind);
    cnnsum_0 = sum(num_cnn_0(ind(1):ind(num)));
    groudsum0 = sum(Num_ground0(ind(1):ind(num)));
    n_cnn_0(num_pati0) = cnnsum_0;
    n_ground0(num_pati0) = groudsum0;
    j = j + num;
    num_pati0 = num_pati0 + 1;
end
R1 = corrcoef(n_cnn_0,n_ground0)
figure;plotregression(n_ground0,n_cnn_0);grid on;
