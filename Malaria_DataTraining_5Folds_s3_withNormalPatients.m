%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program trains on both normal patient and infected patients, performing with 2 steps:
% 1st step: divide the positive and negative patches generated from s1.m
% and s2.m into 80% set1 and 20% set2.
% 2nd step: on set1, use the 5 fold cross validation to train the CNN
% network and save the model with the best performance.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;
t0 = tic;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %patch size = 44*44
imx = 44;
imy = 44;
PatsDatasetPath = 'C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\ProgforBangladeshData\Pats_Bangladesh_Jan2018\';
folder_tr = [PatsDatasetPath,'train_new_fix1_withNormal\'];
folder_te = [PatsDatasetPath,'test_new_fix1_withNormal\GreedyPreselection\'];
% mkdir(folder_tr);
% mkdir(folder_te);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1st step: 
%1)copy the infected patient traing and test dataset into the new
%training and test datasets: train_new_fix1->train_new_fix1_withNormal;
%test_new_fix1->test_new_fix1_withNormal.
%2) divide the normal patients into 80% and 20% into train and test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % path_train = [PatsDatasetPath,'train_new_fix1\'];
% % path_test = [PatsDatasetPath,'test_new_fix1\GreedyPreselection\'];
% % path_nor = [PatsDatasetPath,'NormalPatients\'];
% % status_1 = copyfile(path_train, folder_tr);
% % status_00 = copyfile(path_test,folder_te );
% % %% randomly choose 20% data from the normal patients into the test folder
% % temp_folder0 = dir(path_nor);
% % temp_folder0(1:2) = [];
% % num_folder0 = length(temp_folder0);
% % indperm = randperm(num_folder0,num_folder0);
% % num_te = round(num_folder0 * 0.2);
% % for k = 1:num_folder0
% %     k1 = indperm(k);
% %     if k<num_te+1 %% copy to the test folder
% %         path_te = [path_nor,temp_folder0(k1).name,'\'];
% %         tempf = dir(path_te);
% %         tempf(1:2) = [];
% %         for l =1:length(tempf)
% %            path_te_para = [path_te,tempf(l).name,'\'];
% %           status_1 = copyfile(path_te_para, [folder_te,temp_folder0(k1).name,'\']);        
% % %         path_te_neg = [path_te,tempf(l).name,'\0\'];
% % %         status_1 = copyfile(path_te_para, [folder_te,'\1\',temp_folder0(k1).name,'\',tempf(l).name,'\']);
% % %         status_0 = copyfile(path_te_neg, [folder_te,'\0\',temp_folder0(k1).name,'\',tempf(l).name,'\']);
% %         end
% %     else % copy to the training folder
% %         temp_folder(k-num_te) = temp_folder0(k1);
% %     end
% % end
% % %% divide the training datasets into the randomly chosen 5-folds
% % num_folder = length(temp_folder);
% % indperm0 = randperm(num_folder,num_folder);
% % num_1 = round(num_folder * 0.2);%% when using 5-folder cross validation, num_fo_1 is the number for each folder
% % num_2 = round(num_folder * 0.4);
% % num_3 = round(num_folder * 0.6);
% % num_4 = round(num_folder * 0.8);
% % num_5 = num_folder;
% % for ii = 1:5
% %     subfold_0 = [folder_tr,sprintf('fold_%d',ii),'\0\'];
% %     mkdir(subfold_0);
% %     subfold_1 = [folder_tr,sprintf('fold_%d',ii),'\1\'];
% %     mkdir(subfold_1);
% % end
% % parfor i =1:num_folder
% %     i1 = indperm0(i);
% %     if i<num_1+1
% %         subpath_0 = [folder_tr,'fold_1\0\']; %%% subpath_0 = [PatsDatasetPath,'fold_1\0\',temp_folder(i1).name,'\'];
% %         subpath_1 = [folder_tr,'fold_1\1\'];
% %     elseif i<num_2+1
% %         subpath_0 = [folder_tr,'fold_2\0\'];
% %         subpath_1 = [folder_tr,'fold_2\1\'];
% %     elseif i<num_3+1
% %         subpath_0 = [folder_tr,'fold_3\0\'];
% %         subpath_1 = [folder_tr,'fold_3\1\'];
% %     elseif i<num_4+1
% %         subpath_0 = [folder_tr,'fold_4\0\'];
% %         subpath_1 = [folder_tr,'fold_4\1\'];
% %     else
% %         subpath_0 = [folder_tr,'fold_5\0\'];
% %         subpath_1 = [folder_tr,'fold_5\1\'];
% %     end
% %     path_te = [path_nor,temp_folder(i1).name,'\'];
% %     tempf = dir(path_te);
% %     tempf(1:2) = [];
% %     for j = 1:length(tempf)
% %         path_para = [path_te,tempf(j).name,'\1\'];
% %         path_neg = [path_te,tempf(j).name,'\0\'];
% %         status1 = copyfile(path_para, [subpath_1,temp_folder(i1).name,'\',tempf(j).name,'\']);
% %         status0 = copyfile(path_neg,[subpath_0,temp_folder(i1).name,'\',tempf(j).name,'\']);
% %     end
% % end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2nd step: 5-folder cross estimation of CNN model 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for HN = 1:5
    if HN==1
        loc_te_pos = {[folder_tr,sprintf('fold_%d',HN),'\1\']};
        loc_te_neg = {[folder_tr,sprintf('fold_%d',HN),'\0\']};
        loc_val_pos = {[folder_tr,sprintf('fold_%d',HN+1),'\1\']};
        loc_val_neg = {[folder_tr,sprintf('fold_%d',HN+1),'\0\']};
        for j = 1:3
            loc_tr_pos{j,1} = [folder_tr,sprintf('fold_%d',j+2),'\1\']; %% attention, if use here: "loc_tr_pos(i) = ...", we need to use "datastore({loc_tr_pos,loc_tr_neg},...)" for the datastore.
            loc_tr_neg{j,1} = [folder_tr,sprintf('fold_%d',j+2),'\0\'];
        end
    elseif HN==2
        loc_te_pos = {[folder_tr,sprintf('fold_%d',HN),'\1\']};
        loc_te_neg = {[folder_tr,sprintf('fold_%d',HN),'\0\']};
        loc_val_pos = {[folder_tr,sprintf('fold_%d',HN+1),'\1\']};
        loc_val_neg = {[folder_tr,sprintf('fold_%d',HN+1),'\0\']};
        for j = 1:5
            if j<2
                loc_tr_pos{j,1} = [folder_tr,sprintf('fold_%d',j),'\1\']; %% attention, if use here: "loc_tr_pos{i} = ...", we need to use "datastore([loc_tr_pos,loc_tr_neg],...)" for the datastore.
                loc_tr_neg{j,1} = [folder_tr,sprintf('fold_%d',j),'\0\'];
            elseif j>3
                loc_tr_pos{j-2,1} = [folder_tr,sprintf('fold_%d',j),'\1\']; 
                loc_tr_neg{j-2,1} = [folder_tr,sprintf('fold_%d',j),'\0\'];
            end
        end
    elseif HN==3
        loc_te_pos = {[folder_tr,sprintf('fold_%d',HN),'\1\']};
        loc_te_neg = {[folder_tr,sprintf('fold_%d',HN),'\0\']};
        loc_val_pos = {[folder_tr,sprintf('fold_%d',HN+1),'\1\']};
        loc_val_neg = {[folder_tr,sprintf('fold_%d',HN+1),'\0\']};
        for j = 1:5
            if j<3
                loc_tr_pos{j,1} = [folder_tr,sprintf('fold_%d',j),'\1\']; %% attention, if use here: "loc_tr_pos{i} = ...", we need to use "datastore([loc_tr_pos,loc_tr_neg],...)" for the datastore.
                loc_tr_neg{j,1} = [folder_tr,sprintf('fold_%d',j),'\0\'];
            elseif j>4
                loc_tr_pos{j-2,1} = [folder_tr,sprintf('fold_%d',j),'\1\']; 
                loc_tr_neg{j-2,1} = [folder_tr,sprintf('fold_%d',j),'\0\'];
            end
        end
    elseif HN==4
        loc_te_pos = {[folder_tr,sprintf('fold_%d',HN),'\1\']};
        loc_te_neg = {[folder_tr,sprintf('fold_%d',HN),'\0\']};
        loc_val_pos = {[folder_tr,sprintf('fold_%d',HN+1),'\1\']};
        loc_val_neg = {[folder_tr,sprintf('fold_%d',HN+1),'\0\']};
        for j = 1:3
            loc_tr_pos{j,1} = [folder_tr,sprintf('fold_%d',j),'\1\']; %% attention, if use here: "loc_tr_pos{i} = ...", we need to use "datastore([loc_tr_pos,loc_tr_neg],...)" for the datastore.
            loc_tr_neg{j,1} = [folder_tr,sprintf('fold_%d',j),'\0\'];
        end
    elseif HN==5
        loc_te_pos = {[folder_tr,sprintf('fold_%d',HN),'\1\']};
        loc_te_neg = {[folder_tr,sprintf('fold_%d',HN),'\0\']};
        loc_val_pos = {[folder_tr,sprintf('fold_%d',1),'\1\']};
        loc_val_neg = {[folder_tr,sprintf('fold_%d',1),'\0\']};
        for j = 2:4            
            loc_tr_pos{j-1,1} = [folder_tr,sprintf('fold_%d',j),'\1\']; %% attention, if use here: "loc_tr_pos{i} = ...", we need to use "datastore([loc_tr_pos,loc_tr_neg],...)" for the datastore.
            loc_tr_neg{j-1,1} = [folder_tr,sprintf('fold_%d',j),'\0\'];
        end
    end
    %% training data store
    ParaData_tr = datastore(loc_tr_pos,...
        'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
    ParaData_tr.Labels = categorical(ones(size(ParaData_tr.Files)));
    NegData_tr = datastore(loc_tr_neg,...
        'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
    NegData_tr.Labels = categorical(zeros(size(NegData_tr.Files)));
    trainPatsData = datastore([loc_tr_pos;loc_tr_neg],'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
    trainPatsData.Labels = [ParaData_tr.Labels; NegData_tr.Labels];
    trainPatsData.ReadFcn = @(loc)imresize(imread(loc),[imx imy]);
    % to augment the data:
%     augmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandRotation',[90 90], ...
%     'RandYReflection',true);
%     traindatasource = augmentedImageDatastore([imx imy 3],trainPatsData, 'DataAugmentation',augmenter);
    %%validation data store
    ParaData_val = datastore(loc_val_pos,...
        'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
    ParaData_val.Labels = categorical(ones(size(ParaData_val.Files)));
    NegData_val = datastore(loc_val_neg,...
        'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
    NegData_val.Labels = categorical(zeros(size(NegData_val.Files)));
    valPatsData = datastore([loc_val_pos;loc_val_neg],'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
    valPatsData.Labels = [ParaData_val.Labels; NegData_val.Labels];
    valPatsData.ReadFcn = @(loc)imresize(imread(loc),[imx imy]);
    % to augment the data:
%     valdatasource = augmentedImageDatastore([imx imy 3],valPatsData, 'DataAugmentation',augmenter);
    %%test data store
    ParaData_te = datastore(loc_te_pos,...
        'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
    ParaData_te.Labels = categorical(ones(size(ParaData_te.Files)));
    NegData_te = datastore(loc_te_neg,...
        'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
    NegData_te.Labels = categorical(zeros(size(NegData_te.Files)));
    testPatsData = datastore([loc_te_pos;loc_te_neg],'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
    testPatsData.Labels = [ParaData_te.Labels; NegData_te.Labels];
    testPatsData.ReadFcn = @(loc)imresize(imread(loc),[imx imy]);
    % to augment the data:
%     testdatasource = augmentedImageDatastore([imx imy 3],testPatsData, 'DataAugmentation',augmenter);
    
    %%read the positive and negative patches
%     labelCount_tr = countEachLabel(traindatasource)
%     labelCount_val = countEachLabel(valdatasource)
    labelCount_te = countEachLabel(testPatsData)
    %% data augmentation
    %augmenter = imageDataAugmenter(...
         %'RandXTranslation',[-10 10],...
        %'RandYTranslation',[-10 10]);
    %trainDataSource = augmentedImageSource([imx imy 3],trainPatsData,'DataAugmentation',augmenter);
    %% Define Network Architecture: Define the convolutional neural network architecture.
% %     layers = [
% %         imageInputLayer([imx imy 3])
% %         
% %         convolution2dLayer(5,16,'Padding',1)
% %         batchNormalizationLayer
% %         reluLayer
% %         
% %         maxPooling2dLayer(2,'Stride',2)
% %         
% %         convolution2dLayer(5,32,'Padding',1)
% %         batchNormalizationLayer
% %         reluLayer
% %         
% %         maxPooling2dLayer(3,'Stride',2)
% %         
% %         convolution2dLayer(3,64)
% %         batchNormalizationLayer        
% %         reluLayer
% %         
% % %         fullyConnectedLayer(50) %% dropout layer can be added here to test
% % %         dropoutLayer(0.5)
% % %         fullyConnectedLayer(2)
% %         %%old
% %         fullyConnectedLayer(512) %% dropout layer can be added here to test
% %         fullyConnectedLayer(2)
% %         %%end of old
% %         softmaxLayer
% %         classificationLayer];
    %% define a resnet-similar network
%%this network is completed in Malaria_DataTraining_ResNet9.m
    %% define a deeper CNN for comparison:
    layers = [
        imageInputLayer([imx imy 3])
        
        convolution2dLayer(3,16,'Padding',1) %conv1 (->16 @ 44x44)
        batchNormalizationLayer
        reluLayer
        
        convolution2dLayer(3,16,'Padding',1) %conv2  (->16 @ 44x44)
        batchNormalizationLayer
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)  %   (->16 @ 22x22)
        
        convolution2dLayer(3,32,'Padding',1) %conv3  (->32 @ 22x22)
        batchNormalizationLayer
        reluLayer
        
        convolution2dLayer(3,32,'Padding',1) %conv4  (->32 @ 22x22)
        batchNormalizationLayer
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)   %  (->32 @ 11x11)
        
        convolution2dLayer(3,64,'Padding',1) %conv5  (->64 @ 11x11)
        batchNormalizationLayer
        reluLayer
        
        convolution2dLayer(3,64,'Padding',1) %conv6  (->64 @ 11x11)
        batchNormalizationLayer
        reluLayer
        
        maxPooling2dLayer(3,'Stride',2)    % (->64 @ 5x5)
        
        convolution2dLayer(3,64,'Padding',1) %conv7  (->64 @ 5x5)
        batchNormalizationLayer
        reluLayer
        
        fullyConnectedLayer(512) %% dropout layer can be added here to test
        dropoutLayer(0.5)
        fullyConnectedLayer(50)
        dropoutLayer(0.5)
        fullyConnectedLayer(2)
        
        softmaxLayer
        classificationLayer
        ];

%% plot the 10th layer parameters
% [~,~,iter,~]=size(net.Layers(10).Weights);
% name='weight.gif';
% dt=0.4;
% for i=1%:iter
% 	montage(imresize(mat2gray(net.Layers(10).Weights(:,:,i,:)),[128 128]));
%     set(gcf,'color',[1 1 1]); %to be white
% 	title(['Layer(10), Channel: ',num2str(i)]);
% 	axis normal
% 	truesize
% 	%Creat GIF
% 	frame(i)=getframe(gcf); % get the frame
% 	image=frame(i).cdata;
% 	[image,map]     =  rgb2ind(image,256);  
% 	if i==1
% 		 imwrite(image,map,name,'gif');
% 	else
% 		 imwrite(image,map,name,'gif','WriteMode','append','DelayTime',dt);
%     end
% end
    %% to fix the initialization of the weight of the first convolutional layer:(The default for the initial weights is a Gaussian distribution with a mean of 0 and a standard deviation of 0.01. The default for the initial bias value is 0.):
    % layers(2).Weights=randn([5 5 3 16])*0.0001; % the size of the local regions in the layer is 5-by-5, and the number of color channel for each region is 3. The number of feature maps is 16, so there are 5*5*3*16 weights in this convolutional layer.
    % layers(2).Bias = randn([1 1 16])*0.00001 + 1;% there are 16 feature maps, and therefore 16 bias.
    %% to fix the initialization of the fully connected layer:
    % layers(13).Weights = randn([512 3136])*0.0001; %% 3136 is the input size of the fully connection layer: 62@7*7
    %%
    options = trainingOptions('sgdm',...
        'MaxEpochs',20, ...
        'ValidationData',valdatasource,...
        'ValidationFrequency',30,...
        'Shuffle','every-epoch',...%new added by feng 03/03/2018
        'InitialLearnRate',0.001,...
        'Verbose',true);
        %'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3) );
        %'CheckpointPath', 'C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\ProgforBangladeshData\checkpoint\');
    %'Plots','training-progress');
    [net(HN),tr(HN)] = trainNetwork(traindatasource,layers,options);
    %[net(HN),tr(HN)] = trainNetwork(trainDataSource,layers,options);
    [predictedLabels,scores] = classify(net(HN),testPatsData,'ExecutionEnvironment','gpu');
    valLabels = testPatsData.Labels;
    accuracy(HN) = sum(predictedLabels == valLabels)/numel(valLabels)
    toc(t0)
    % disp(['Estimate the running time for training patches: ',num2str(etime(clock,t2))]);
    %% computing the confusion table
    g1 = zeros(size(valLabels,1),size(valLabels,2));
    g1(valLabels=='1')=1;
    g2 = zeros(size(valLabels,1),size(valLabels,2));
    g2(predictedLabels=='1')=1;
    % [C, order] = confusionmat(g1,g2)
    figure;plotconfusion(g1',g2')
    sensi(HN) = sum(predictedLabels == valLabels & valLabels=='1')/sum(valLabels=='1');% is also called recall
    speci(HN) = sum(predictedLabels == valLabels & valLabels=='0')/sum(valLabels=='0');
    precision(HN) = sum(predictedLabels == valLabels & valLabels=='1')/sum(predictedLabels=='1');% is also called positive prediciton ratio
    negpred(HN) = sum(predictedLabels == valLabels & valLabels=='0')/sum(predictedLabels=='0');
    f_score(HN) = 2*sensi(HN)*precision(HN)/(sensi(HN)+precision(HN))
    %% compute the ROC and AUC
    [X,Y,T,AUC] = perfcurve(valLabels,scores(:,1),'1') ;
    AUC_a(HN) = AUC
    figure(2);plot(X,Y,'r','LineWidth',2)
    xlabel('False positive rate')
    ylabel('True positive rate')
    title('ROC for Parasiste Classification')
    hold on;
    %% to plot the number of "1" along with the threshold
% %     predictedLabels1 = predictedLabels;
% %     num = 1;
% %     num_pos(num) = length( find(predictedLabels=='1' ) );
% %     for num_th = 0.6:0.1:1
% %         num = num + 1;
% %         predictedLabels = predictedLabels1;
% %         predictedLabels(scores(:,1)<num_th) = categorical(0);
% %         num_pos(num) = length( find(predictedLabels=='1' ) );
% %     end
% %     xx = 0.5:0.1:1;
% %     figure(3); plot(xx,num_pos);
end
legend('Fold1','Fold2','Fold3','Fold4','Fold5');
num_f = find(accuracy==max(accuracy(:)))
net_1run = net(num_f);
%% save accuracy and AUC info
pathname = ['C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\ProgforBangladeshData\Results\',sprintf('train_new_fix1_withnormal_pat%d',imx),'\'];
mkdir(pathname);
filename2 = [pathname,sprintf('AccuracyInfos_Val_train5Folds_withNormal_%d',imx),'.txt'];
file2 = fopen(filename2,'w+');
fprintf(file2,'%s %8.4f %8.4f %8.4f %8.4f %8.4f\n','Accuracy=',accuracy(1), accuracy(2), accuracy(3),accuracy(4),accuracy(5));
fprintf(file2,'%s %8.4f %8.4f %8.4f %8.4f %8.4f\n','F_socre=',f_score(1), f_score(2), f_score(3),f_score(4),f_score(5));
fprintf(file2,'%s %8.4f %8.4f %8.4f %8.4f %8.4f\n','AUC=',AUC_a(1), AUC_a(2), AUC_a(3),AUC_a(4),AUC_a(5));
fprintf(file2,'%s %8.4f %8.4f %8.4f %8.4f %8.4f\n','sensi=',sensi(1), sensi(2), sensi(3),sensi(4),sensi(5));
fprintf(file2,'%s %8.4f %8.4f %8.4f %8.4f %8.4f\n','speci=',speci(1), speci(2), speci(3),speci(4),speci(5));
fprintf(file2,'%s %8.4f %8.4f %8.4f %8.4f %8.4f\n','precision=',precision(1), precision(2), precision(3),precision(4),precision(5));
fprintf(file2,'%s %8.4f %8.4f %8.4f %8.4f %8.4f\n','negpred=',negpred(1), negpred(2), negpred(3),negpred(4),negpred(5));
fprintf(file2,'%s %8.4f\n','avg_accuracy=',mean(accuracy));
fprintf(file2,'%s %8.4f\n','avg_AUC=',mean(AUC_a));
fclose(file2);

%% train on more data and get the final network for Set2
loc_val_pos = {[folder_tr,sprintf('fold_%d',1),'\1\']};
loc_val_neg = {[folder_tr,sprintf('fold_%d',1),'\0\']};
for j = 1:4
    loc_tr_pos{j,1} = [folder_tr,sprintf('fold_%d',j+1),'\1\']; %% attention, if use here: "loc_tr_pos(i) = ...", we need to use "datastore({loc_tr_pos,loc_tr_neg},...)" for the datastore.
    loc_tr_neg{j,1} = [folder_tr,sprintf('fold_%d',j+1),'\0\'];
end
%%train data
ParaData_tr = datastore(loc_tr_pos,...
    'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
ParaData_tr.Labels = categorical(ones(size(ParaData_tr.Files)));
NegData_tr = datastore(loc_tr_neg,...
    'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
NegData_tr.Labels = categorical(zeros(size(NegData_tr.Files)));
trainPatsData = datastore([loc_tr_pos;loc_tr_neg],'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
trainPatsData.Labels = [ParaData_tr.Labels; NegData_tr.Labels];
trainPatsData.ReadFcn = @(loc)imresize(imread(loc),[imx imy]);
% traindatasource = augmentedImageDatastore([imx imy 3],trainPatsData, 'DataAugmentation',augmenter);
ParaData_val = datastore(loc_val_pos,...
    'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
ParaData_val.Labels = categorical(ones(size(ParaData_val.Files)));
NegData_val = datastore(loc_val_neg,...
    'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
NegData_val.Labels = categorical(zeros(size(NegData_val.Files)));
valPatsData = datastore([loc_val_pos;loc_val_neg],'IncludeSubfolders',true,'FileExtensions','.png','Type','image');
valPatsData.Labels = [ParaData_val.Labels; NegData_val.Labels];
valPatsData.ReadFcn = @(loc)imresize(imread(loc),[imx imy]);
valdatasource = augmentedImageDatastore([imx imy 3],valPatsData, 'DataAugmentation',augmenter);
labelCount_tr = countEachLabel(trainPatsData)
labelCount_val = countEachLabel(valPatsData)
[net_new,tr] = trainNetwork(trainPatsData,layers,options);%[net_new,tr] = trainNetwork(traindatasource,layers,options);
save net_train_new_fix1_withNormal_drop_2_05_val.mat net_new;