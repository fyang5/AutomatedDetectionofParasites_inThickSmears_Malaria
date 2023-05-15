%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program is to count the WBC and parasite numbers according to the
% annotation files (ground truth)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;
%%paths for training network
pat0_path = 'C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\ProgforBangladeshData\Pats_Bangladesh\randImg\0\';
pat1_path = 'C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\ProgforBangladeshData\Pats_Bangladesh\randImg\1\';
% mydir = dir('C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\ProgforBangladeshData\Pats_Bangladesh\randImg\0');
% delete *.png;
t0=tic;
%% test an random image without annotation
s1 = 44;
s2 = 44;
dfactor = 30
ratio_th = 0.1
%%path for image
pathname = 'C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\Bangladesh_may_2017\';

%%path for annotation files
path_folderanno = 'C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\Bangladesh_may_2017\GT\';
temp_folder = dir(path_folderanno);
%% remove the normal patient folders
% temp_folder(119)=[];
% temp_folder(108:117)=[];
% temp_folder(97:106)=[];
% temp_folder(86:95)=[];
% temp_folder(75:84)=[];
% temp_folder(65:73)=[];
%%end of removing of the normal patient folders
%%
num_folder = length(temp_folder);
filename2 = 'WBCandParasiteGroundTruthforAllPatients_withFrameNo_new.txt';
file2 = fopen(filename2,'w+');
%% choose a random folder and a random file for annotation, and then read the corresponding image
nn=0;
for indperm0 = 3:num_folder
    % indperm0 = randperm(num_folder,1)
    foldername = temp_folder(indperm0).name;
    pathname_anno = [path_folderanno, temp_folder(indperm0).name,'\'];
    temp_anno = dir([pathname_anno,'*.txt']);
    num_anno = length(temp_anno);
    num_anno_p = 0;
    num_anno_z = 0;
    %%re-organize the annotation files in this folders, and remove the files without annotations
    for i1 = 1:num_anno
        filename = temp_anno(i1).name;
        file_anno = [pathname_anno, temp_anno(i1).name];
        fid = fopen(file_anno);
        s = textscan(fgetl(fid),'%s','delimiter',',');
        s=s{1};
        Nlab = str2num(s{1});
        if Nlab~=0
            num_anno_p = num_anno_p + 1;
            files_anno{num_anno_p} = filename(1:end-4);
        else
            num_anno_z = num_anno_z + 1;
            files_anno_n{num_anno_z} = filename(1:end-4);
        end
        fclose(fid);
    end
    if num_anno_z ~= num_anno %% "num_anno_z==num_anno" means all the annotation files have only one line without annotation, so it is uninfected patient
        files_anno(num_anno_p+1:end) = [];
    else
        files_anno_n(num_anno_z+1:end) = [];
        files_anno = files_anno_n;
    end
    for indperm = 1:length(files_anno)
        % indperm = randperm(length(files_anno), 1);%14;   %%% randomly choose a file
        filename_anno = files_anno{indperm}; %% annotation file name
        file_anno = [pathname_anno, filename_anno,'.txt']; %% annotation file name and directory
        %%read corresponding image
        pathname_img = [pathname, temp_folder(indperm0).name,'\'];
        file_img = [pathname_img, filename_anno,'.jpg'];
        im0 = imread(file_img);
        im = rgb2gray(im0);
        [GT,lbls,mask,pos,num_para,Nlab] = loadGT_new_1(file_anno,s1);
        %% read data.txt in the folder to get the FrameNo in log file of Oye
        dataFile = [pathname_img, 'data.txt'];
        fid1 = fopen(dataFile,'r');
        a1 = textscan(fid1,'%s');
        fclose(fid1);
        fileNo = a1{1,1};
        tname = [filename_anno,'.jpg'];
        fNo = find(ismember(fileNo,tname))-1; %% here we cannot use strcmp(fileNo,tname),since fileNo is a cell array
        %% exclude the files which are useless: Nlab = 0 means no annotation are performed
        if Nlab~=0
            fprintf(file2,'%s %s %d %d %d\n',foldername, filename_anno,fNo,num_para,Nlab-num_para);
        end
        
%         if Nlab~=0
%             nn=nn+1;
%             % figure(1); imshow(im0), title('Thick parasite smear');
%             % text(size(im0,2),size(im0,1)+15,...
%             %     ['the number of the image is:', num2str(indperm)], ...
%             %     'FontSize',7,'HorizontalAlignment','right');
%             % [imx, imy, n] = size(im0);
%             
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %% 1st step: WBC dectection and removing
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Radius=20;%??????????????
%             RBC_AvgArea=pi*((Radius)^2);
%             %% ROI Extraction
%             Image = im0;
%             cform = makecform('srgb2cmyk');
%             cmykI = applycform(Image,cform);
%             Yellow=cmykI(:,:,3); % Yellow channel
%             H=1-imbinarize(Yellow);
%             H = bwareaopen(H, 50);
%             H=imfill(H,'holes');
%             [limit_row,limit_col]=find(H~=0);
%             
%             % Yellow=Yellow(min(limit_row)+5:max(limit_row)-5,min(limit_col)+5:max(limit_col)-5);
%             % Image=Image(min(limit_row)+5:max(limit_row)-5,min(limit_col)+5:max(limit_col)-5,1:3);
%             % %GTMask=GTMask(min(limit_row)+5:max(limit_row)-5,min(limit_col)+5:max(limit_col)-5);
%             % H=H(min(limit_row)+5:max(limit_row)-5,min(limit_col)+5:max(limit_col)-5);
%             
%             num_rows=size(Image,1);
%             num_cols=size(Image,2);
%             
%             for i=1:3
%                 channel=Image(:,:,i);
%                 channel(find(H==0))=0;
%                 Image(:,:,i)=channel;
%             end
%             
%             %% Method 2: selete and remove the white blood cell candidates
%             img = rgb2gray(Image); %t = multithresh(img);img(img>t)=255; img1=1-imbinarize(img); %% Using global threshold to binarize the image
%             img1=1-imbinarize(img,graythresh(img)); %% Using global threshold Otsu methodto binarize the image
%             H=imerode(H,strel('disk',15)); %% erode the ROI area == dilate the background area
%             img1(H==0)=0; % remove the background area
%             img2= bwareaopen(img1,700); %% delete the small areas
%             img2 = imdilate(img2,strel('disk',18));%% dilate the small roi
%             img2 = imfill(img2,'holes');%% fill in the area
%             stats2 = regionprops('table',img2, 'Area','Centroid','MajorAxisLength','MinorAxisLength','BoundingBox');
%             allArea2 = stats2.Area;
%             num = 0;
%             %figure(1);imshow(Image);
%             for k= 1:length(allArea2)
%                 if allArea2(k)<1.0e5 && allArea2(k)> 1000
% % %                     cx =  round(stats2.Centroid(k,1)) ; %% the center of the circle or rectangle
% % %                     cy =  round(stats2.Centroid(k,2)) ;
% % %                     rad = round(stats2.MajorAxisLength(k));
% % %                     cent = [cx, cy];
% % %                     figure(1); hold on;viscircles(cent,rad/2,'LineWidth',1);
% % %                     wid = round(rad);
% % %                     hei = round(rad);
%                     %figure(1);hold on;rectangle('Position',[cx-wid/2+1 cy-hei/2+1 wid hei],'EdgeColor','r','LineWidth',1);
%                     num = num + round(allArea2(k)/median(allArea2));
%                     %dilBW1(cy-round(hei/2)+1:cy+round(hei/2),cx-round(wid/2)+1:cx+round(wid/2) )=0;
%                 end
%             end
%             %%disp(['The candidates of the white blood cells is: ', num2str(num)]);
%             %%Evaluation by the ground truth
% %             [GT,lbls,mask,pos,num_para,Nlab] = loadGT_new_1(file_anno,s1);
%             %%disp(['The ground truth of the white blood cells is: ', num2str(Nlab-num_para)]);
%              wbcAll(nn,1:2)=[num Nlab-num_para]';
%              % % % % % %               wbcAll(nn,1:2)=Nlab-num_para';
%              fprintf(file2,'%s %s %d %d\n',foldername, filename_anno,num, Nlab-num_para);
%              %             %             figure(2); subplot(222);imshow(img1);title('threshold-segmented WBCs');
%              %             %             subplot(221);imshow(Image);title('Original Image');
%              %             %             subplot(223);imshow(img2);title('filtered WBC areas-threshold');
%              %             %             subplot(224);imshow(RMask);title('Range filtered WBCs');
%              %             for i=1:3
%              %                 % find the mean of the non-zero area of Image
%              %                 channel1 = Image(:,:,i);
%              %                 channel1(find(img2==1 | H==0))=[];
%              %                 avg = median(channel1);
%              %                 % let the WBCs area equals to avg or 0;
%              %                 channel=Image(:,:,i);
%              %                 channel(find(img2==1))= 0;
%              %                 Image(:,:,i)=channel;
%              %             end
%              %             % figure(2);subplot(224);imshow(Image);title('Image after removing WBCs');
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %% 2nd step: preselection of the parasite candidates
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %             %% method 1: preselection using the sobel filter for edge detection
%             %             im = Image(:, :, 2);
%             %             BW1 = edge(im,'Sobel');
%             %             BW1(H==0) = 0; %% using the H image to remove the background
%             %             BW1 = bwareaopen(BW1,8);se1 = strel('disk',2);dilBW1 = imdilate(BW1,se1);%figure(4);imshow(dilBW1)
%             %             stats1 = regionprops('table',dilBW1, 'Area','Centroid','MajorAxisLength','MinorAxisLength','BoundingBox');
%             %             dilBW1(img2==1) = 0;
%             %             % % figure(5);imshow(dilBW1);title('Sobel filter preselection')
%             %             %% generate patches according to the selection
%             %             stats = regionprops('table',dilBW1, 'Area','Centroid','MajorAxisLength','MinorAxisLength','BoundingBox');
%             %             AllBoundBox = stats.BoundingBox;
%             %             X0 = round(AllBoundBox);
%             %             % % npat = 0;
%             %             % % npat_p = 0;
%             %             % % npat_n = 0;
%             %             % % npat_gr = 0;
%             %             % ground truth of parasite patches
%             %             fprintf(file1,'%s %s %d %d\n',foldername, filename_anno,length(AllBoundBox),num_para);
%             %
%             %             len(nn,1:2)=[length(AllBoundBox) num_para]';
%         end
    end
end
% % fprintf(file2,'\n%d %d %d %d\n',num_folder-2, nn, mean(wbcAll(:,1)), mean(wbcAll(:,2)) );
fprintf(file2,'\n%s %s %s %s %s\n','foldername', 'filename','frameNo', 'ParasiteNum', 'WBCNum' );
fclose(file2);
%% plot the curve of estimated WBC numbers

% % x=1:length(wbcAll);
% % y1 = wbcAll(:,1);
% % figure;
% % y2 = wbcAll(:,2);
% % % plot(x,y2,'-*r','LineWidth',2);
% % % hold on;
% % plot(x,y1-y2,':ob','LineWidth',2);
% % xlabel('Image number');
% % ylabel('Different number of WBCs between our estimation and ground truth');
% % grid on;
% % legend('EstimatedWBCs-ground truth')