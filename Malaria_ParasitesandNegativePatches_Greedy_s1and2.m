%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program equals to the 1st and 2nd steps to prepare our patches for
%   CNN model. In this program, we obtain both positive(parasites) and 
%   negative patches using the GREEDY method.  The patches
%   are saved with a fixed size-- a rectangle region with the width
%   and height are both 2*22, for which only the information within the
%   circle (center: [cx, cy], radius: rad) are preserved, and the other
%   area inside this rectangle are saved as [0 0 0]. For infected patients,
%   patches are saved separately in terms of the name of
%   "patient_name/annotation_file_name". A *.txt file is also saved
%   in this folder to indicates the number of parasites, White Blood Cells,
%    mean of the radius of annotated circles, etc.
%
%   temp_folder: a cell array which saves all the annotation folders
%
%    Feng Yang
%    NIH/NLM/CEB
%    December 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% In this program, we just obtain the parasite patches with their original size.
% For infected patients, parasite patches are saved separately in terms of
% the name of "patient_name/annotation_file_name". A *.txt file is also saved
% in this folder to indicates the number of parasites, White Blood Cells,
% mean of the radius of annotated circles, etc.

clear all; clc; close all;
s1 = 44;
s2 = 44;
rad = s1*0.5;
ratio_th = 0.5
% nover = s1*s2*ratio_th
num_th = 500;
tic; t1 = clock;
% %define the padding size for the images, note that here s1 and s2 have
% different meaning for traning data
s1 = 100;% here is just for padding the image
s2 = 100;
% parasite patch directory: top_folder\0or1\patient_folder\image_folder\
top_folder = 'C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\ProgforBangladeshData\Pats_Bangladesh_Jan2018\Greedy\';
%%image folder directory
pathname = 'C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\Bangladesh_may_2017\';
%%annotation folder directory
path_folderanno = 'C:\Users\yangf5\Desktop\yf\Work-nih\Malaria\MatlabProg\Bangladesh_may_2017\GT_updated\';
% creat directory to save positive patches
pat_folder_pos = sprintf('%s%s',top_folder,'1\');
mkdir(pat_folder_pos);
pat_folder_neg = sprintf('%s%s',top_folder,'0\');
mkdir(pat_folder_neg);
%% read the annotation folders
temp_folder = dir(path_folderanno);
% only for the infected patients
temp_folder(119)=[];
temp_folder(108:117)=[];
temp_folder(97:106)=[];
temp_folder(86:95)=[];
temp_folder(75:84)=[];
temp_folder(65:73)=[];
%end for the infected patients
temp_folder(1:2) = [];
num_folder = length(temp_folder);
num_pat = 0;
filename2 = [top_folder,'StatisticsforInfectedPatients_03072018.txt'];
file2 = fopen(filename2,'w+');
fprintf(file2,'%s %s %s %s %s %s\n','PatientID', 'FileID','NameofParasite','Radius');
filename3 = [top_folder,'ParasiteSensivitybyGreedy_noverNew.txt.txt'];
file3 = fopen(filename3,'w+');
fprintf(file3,'%s %s %s %s %s %s \n','foldername', 'filename_anno','num_para','num_overlay','para_preselection','neg_num' );

for j1 = 11%1:num_folder %% the first two folders are useless, with '.' and '..'
    foldername = temp_folder(j1).name;
    pathname_anno = [path_folderanno, temp_folder(j1).name,'\'];
    temp_anno = dir([pathname_anno,'*.txt']);
    num_anno = length(temp_anno);
    num_anno_p = 0; %% the number of annotated files
    %% re-organize the annotation files in this folders, and remove the files without annotations
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
        end
        fclose(fid);
    end
    %% read the images according to the annotation file name
    files_anno(num_anno_p+1:end) = []; %% clear all the elements after num_anno_p
    pathname_img = [pathname, temp_folder(j1).name,'\'];
    % creat corresponding patient folder for saving parasite patches
    patient_folder1 = sprintf('%s%s',pat_folder_pos,foldername);
    mkdir(patient_folder1);
    patient_folder0 = sprintf('%s%s',pat_folder_neg,foldername);
    mkdir(patient_folder0);
    %%
    num_aWBC = 0;%% this is the number of files which only contains white blood cell annotation without parasites.
    for i1 = 1%:length(files_anno)
        %% read the annotation file and corresponding image file
        filename_anno = files_anno{i1}; %% annotation file name
        file_anno = [pathname_anno, filename_anno,'.txt']; %% annotation file name and directory
        file_img = [pathname_img, filename_anno,'.jpg'];
        im0 = imread(file_img);
        [imy,imx,nc] = size(im0);
        [GT,lbls,mask_anno,para_pos,num_para,Nlab] = loadGT_new_1(file_anno,s1);
        if num_para~=0
            %%creat folder for each image to save parasite patches
            image_folder = sprintf('%s%s%s',patient_folder1,'\',filename_anno);
            mkdir(image_folder);
            path_pat1 = sprintf('%s%s',image_folder,'\');%% path name to save the patches of current image
            %for negative patches:
            image_folder0 = sprintf('%s%s%s',patient_folder0,'\',filename_anno);
            mkdir(image_folder0);
            path_pat0 = sprintf('%s%s',image_folder0,'\');
            %pad image and masks from annotations
            im0 = padarray(im0,[s1 s2],'post'); %% pad zeros at the end of im in both x and y direction
            mask_anno = padarray(mask_anno,[s1 s2],'post');            
            mask_anno1=mask_anno;
            lbls = padarray(lbls,[s1 s2],'post');
        [image_anno,Markers,Marker_colors,circle_shapes,circle_colors,polygon_shapes,cell_type] = AnnotateImage(im0,file_anno); % Read GT.txt file and parse marker/region-wise annotations
        figure(1);imshow(image_anno);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% greedy method: find the lowest intensity to generate positive and negative patches
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [border_mask,WBC_mask]= Compute_border_mask_new(im0);
            candi = double(rgb2gray(im0));%im0(:,:,2)
            %%candi(mask_anno~=0) = 0; %% if this script is used, it means we only choose negative patches
            candi = candi.*(1-border_mask);%figure;imshow(candi);
            candi1 = candi; %% candi1 includes the WBCs
            candi1(mask_anno~=0) = 0;%candi1 indicates the area where there is no parasites
            candi(WBC_mask==1) = 0;
            candi_temp = candi;
            candi_temp(candi_temp==0) = [];
            min_temp = min(candi_temp(:));
            candi_mask = zeros(size(candi,1),size(candi,2));
            mask_cmp = zeros(size(mask_anno,1),size(mask_anno,2));% indicate the mask of parasites that is generately by greedy method
            num_neg = 0;
            num_pos = 0;
            num_pat_n = 0;
            while num_pat_n < num_th
                [col,row] = find(candi==min_temp);
                for i = 1 : size(col,1)
                    cen_y = col(i);
                    cen_x = row(i);
                    if i==1
                        [mask_3d,mask_cir,candi_mask] = circle_ROI (candi_mask,rad,cen_x,cen_y);
                        overlay1 = mask_anno.*mask_cir; % indicate the overmap between current circle and mask_anno (mask_anno represents the parasite radius) 
                        overlay2 = lbls.*mask_cir; %indicate the overmap between current circle and lbls(lbls represents the annotation index)
                        overlay0 = candi1.*mask_cir;
                        if length(find(overlay2~=0))~=0
                            over_temp = overlay2;
                            over_temp(over_temp==0)=[];
                            ind1 = max(over_temp);
                            ind2 = min(over_temp);
                            rad1 = sum(overlay1(overlay2==ind1))/length(find(overlay2==ind1))
                            rad2 = sum(overlay1(overlay2==ind2))/length(find(overlay2==ind2))
                            n_ind1 = length(find(overlay2==ind1))/(pi*min(rad1,rad)*min(rad1,rad))
                            n_ind2 = length(find(overlay2==ind2))/(pi*min(rad2,rad)*min(rad2,rad))  
                            if n_ind1>=n_ind2
%                                 rad1 = sum(overlay1(overlay2==ind1))/n_ind1
%                                 if rad1~=0 && rad1>rad
%                                     nover = pi*rad*rad*ratio_th
%                                 elseif rad1~=0 && rad1<=rad
%                                     nover = pi*rad1*rad1*ratio_th
%                                 end
                                
                                if n_ind1>ratio_th && cen_y>rad-1 && cen_x>rad-1
                                    candi_mask(overlay2==ind2) = 0; %in this case, ind2 parasite is not considered here
                                    num_pos = num_pos + 1;
                                    num_pat_n = num_pat_n + 1;
                                    yf1(num_pos) = rad1;
                                    im1 = uint8( double(im0).* mask_3d );
                                    pat1 = im1(cen_y-rad+1:cen_y+rad,cen_x-rad+1:cen_x+rad,:);
                                    pat_name = [path_pat1,sprintf('%s',num2str(num_pat_n),'.png')];
                                    imwrite(pat1,pat_name);
                                    fprintf(file2,'%s %s %d %d %d %8.4f\n',foldername, filename_anno,num_pat,rad1);
                                    figure(1);cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','r','LineWidth',1.5);
                                    mask_anno(overlay2==ind1)=0;
                                    lbls(overlay2==ind1)=0;
                                    %mask_cir(overlay2==ind2)= 0;%% remove the ind2 parasite in mask_cir
                                    mask_cmp = mask_cmp + mask_cir;
                                    if num_pat_n==num_th
                                        break;
                                    end
                                    
                                else
                                    figure(1);cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','b','LineWidth',1.5);
                                end
                            else
%                                 rad1 = sum(overlay1(overlay2==ind2))/n_ind2
%                                 if rad1~=0 && rad1>rad
%                                     nover = pi*rad*rad*ratio_th
%                                 elseif rad1~=0 && rad1<=rad
%                                     nover = pi*rad1*rad1*ratio_th
%                                 end
                                
                                
                                if n_ind2>ratio_th && cen_y>rad-1 && cen_x>rad-1
                                    candi_mask(overlay2==ind1) = 0; %in this case, ind1 parasite is not considered here
                                    num_pos = num_pos + 1;
                                    num_pat_n = num_pat_n + 1;
                                    yf1(num_pos) = rad2;
                                    im1 = uint8( double(im0).* mask_3d );
                                    pat1 = im1(cen_y-rad+1:cen_y+rad,cen_x-rad+1:cen_x+rad,:);
                                    pat_name = [path_pat1,sprintf('%s',num2str(num_pat_n),'.png')];
                                    imwrite(pat1,pat_name);
                                    fprintf(file2,'%s %s %d %d %d %8.4f\n',foldername, filename_anno,num_pat,rad2);
                                    figure(1);cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','r','LineWidth',1.5);
                                    mask_anno(overlay2==ind2)=0;
                                    lbls(overlay2==ind2)=0;
                                    %mask_cir(overlay2==ind1)= 0;%% remove the ind1 parasite in mask_cir
                                    mask_cmp = mask_cmp + mask_cir;
                                    if num_pat_n==num_th
                                        break;
                                    end
                                    
                                else
                                    figure(1);cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','b','LineWidth',1.5);
                                end
                            end
                        elseif length(find(mask_cir~=0))==length(find(overlay0~=0))&& cen_y>rad-1 && cen_x>rad-1
                            num_neg = num_neg + 1;
                            num_pat_n = num_pat_n + 1;
                            im1 = uint8( double(im0).* mask_3d );
                            pat1 = im1(cen_y-rad+1:cen_y+rad,cen_x-rad+1:cen_x+rad,:);
                            pat_name = [path_pat0,sprintf('%s',num2str(num_pat_n),'.png')];
                            imwrite(pat1,pat_name);
                            figure(1); cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','g','LineWidth',1);
                            if num_pat_n==num_th
                                break;
                            end
                        
                        end
                    else
                        dis = [cen_y-col(1:i-1), cen_x-row(1:i-1)];
                        dis_norm = zeros(size(dis,1),1);
                        for k = 1:size(dis,1)
                            dis_temp = dis(k,:);
                            dis_norm(k) = norm(dis_temp);
                        end
                        if length( find ( dis_norm < rad ) )==0
                            [mask_3d,mask_cir,candi_mask] = circle_ROI (candi_mask,rad,cen_x,cen_y);
                            overlay1 = mask_anno.*mask_cir;
                            overlay2 = lbls.*mask_cir; %indicate the overmap between current circle and lbls(lbls represents the annotation index)
                            overlay0 = candi1.*mask_cir;
                            if length(find(overlay2~=0))~=0
                                over_temp = overlay2;
                                over_temp(over_temp==0)=[];
                                ind1 = max(over_temp);
                                ind2 = min(over_temp);
                                rad1 = sum(overlay1(overlay2==ind1))/length(find(overlay2==ind1))
                                rad2 = sum(overlay1(overlay2==ind2))/length(find(overlay2==ind2))
                                n_ind1 = length(find(overlay2==ind1))/(pi*min(rad1,rad)*min(rad1,rad))
                                n_ind2 = length(find(overlay2==ind2))/(pi*min(rad2,rad)*min(rad2,rad))                                  
                                if n_ind1>=n_ind2
%                                     rad1 = sum(overlay1(overlay2==ind1))/n_ind1
%                                     if rad1~=0 && rad1>rad
%                                         nover = pi*rad*rad*ratio_th
%                                     elseif rad1~=0 && rad1<=rad
%                                         nover = pi*rad1*rad1*ratio_th
%                                     end
                                    
                                    %if length(find(overlay1==rad1))>=nover && cen_y>rad-1 && cen_x>rad-1
                                    if n_ind1>=ratio_th && cen_y>rad-1 && cen_x>rad-1
                                        candi_mask(overlay2==ind2) = 0; %in this case, ind2 parasite is not considered here
                                        num_pos = num_pos + 1;
                                        num_pat_n = num_pat_n + 1;
                                        yf1(num_pos) = rad1;
                                        im1 = uint8( double(im0).* mask_3d );
                                        pat1 = im1(cen_y-rad+1:cen_y+rad,cen_x-rad+1:cen_x+rad,:);
                                        pat_name = [path_pat1,sprintf('%s',num2str(num_pat_n),'.png')];
                                        imwrite(pat1,pat_name);
                                        fprintf(file2,'%s %s %d %d %d %8.4f\n',foldername, filename_anno,num_pat,rad1);
                                        figure(1);cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','r','LineWidth',1.5);
                                        mask_anno(overlay2==ind1)=0;
                                        lbls(overlay2==ind1)=0;
                                        %mask_cir(overlay2==ind2)= 0;%% remove the ind2 parasite in mask_cir
                                        mask_cmp = mask_cmp + mask_cir;
                                        if num_pat_n==num_th
                                            break;
                                        end
                                        
                                    else
                                        figure(1);cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','b','LineWidth',1.5);
                                    end
                                else
%                                     rad1 = sum(overlay1(overlay2==ind2))/n_ind2
%                                     if rad1~=0 && rad1>rad
%                                         nover = pi*rad*rad*ratio_th
%                                     elseif rad1~=0 && rad1<=rad
%                                         nover = pi*rad1*rad1*ratio_th
%                                     end
                                    
                                    
                                    if n_ind2>=ratio_th && cen_y>rad-1 && cen_x>rad-1
                                        candi_mask(overlay2==ind1) = 0; %in this case, ind1 parasite is not considered here
                                        num_pos = num_pos + 1;
                                        num_pat_n = num_pat_n + 1;
                                        yf1(num_pos) = rad2;
                                        im1 = uint8( double(im0).* mask_3d );
                                        pat1 = im1(cen_y-rad+1:cen_y+rad,cen_x-rad+1:cen_x+rad,:);
                                        pat_name = [path_pat1,sprintf('%s',num2str(num_pat_n),'.png')];
                                        imwrite(pat1,pat_name);
                                        fprintf(file2,'%s %s %d %d %d %8.4f\n',foldername, filename_anno,num_pat,rad2);
                                        figure(1);cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','r','LineWidth',1.5);
                                        mask_anno(overlay2==ind2)=0;
                                        lbls(overlay2==ind2)=0;
                                        %mask_cir(overlay2==ind1)= 0;%% remove the ind1 parasite in mask_cir
                                        mask_cmp = mask_cmp + mask_cir;
                                        if num_pat_n==num_th
                                            break;
                                        end
                                        
                                    else
                                        figure(1);cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','b','LineWidth',1.5);
                                    end
                                end
                            elseif length(find(mask_cir~=0))==length(find(overlay0~=0))&& cen_y>rad-1 && cen_x>rad-1
                                num_neg = num_neg + 1;
                                num_pat_n = num_pat_n + 1;
                                im1 = uint8( double(im0).* mask_3d );
                                pat1 = im1(cen_y-rad+1:cen_y+rad,cen_x-rad+1:cen_x+rad,:);
                                pat_name = [path_pat0,sprintf('%s',num2str(num_pat_n),'.png')];
                                imwrite(pat1,pat_name);
                                figure(1); cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','g','LineWidth',1);
                                if num_pat_n==num_th
                                    break;
                                end
                            end
                        end
                        
                    end
                        
                        
                        
%                         if length(find(mask_cir~=0))&& length(find(overlay1==rad1))>nover && cen_y>rad-1 && cen_x>rad-1
%                             num_pos = num_pos + 1;
%                             num_pat_n = num_pat_n + 1;  
%                             yf1(num_pos) = rad1;
%                                                     im1 = uint8( double(im0).* mask_3d );
%                                                     pat1 = im1(cen_y-rad+1:cen_y+rad,cen_x-rad+1:cen_x+rad,:);
%                                                     pat_name = [path_pat1,sprintf('%s',num2str(num_pat_n),'.png')];
%                                                     imwrite(pat1,pat_name);
%                             fprintf(file2,'%s %s %d %d %d %8.4f\n',foldername, filename_anno,num_pat,rad1);
%                             figure(1);cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','r','LineWidth',1.5);
%                             mask_anno(mask_cir~=0)=0;
%                             mask_cmp = mask_cmp + mask_cir;
%                             if num_pat_n==num_th
%                                 break;
%                             end
%                         elseif length(find(mask_cir~=0))==length(find(overlay0~=0))&& cen_y>rad-1 && cen_x>rad-1
%                             num_neg = num_neg + 1;
%                             num_pat_n = num_pat_n + 1;
%                                                     im1 = uint8( double(im0).* mask_3d );
%                                                     pat1 = im1(cen_y-rad+1:cen_y+rad,cen_x-rad+1:cen_x+rad,:);
%                                                     pat_name = [path_pat0,sprintf('%s',num2str(num_pat_n),'.png')];
%                                                     imwrite(pat1,pat_name);
%                             figure(1); cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','g','LineWidth',1);
%                             if num_pat_n==num_th
%                                 break;
%                             end
%                         else
%                             candi_mask(mask_cir~=0) = 0;
%                             [mask_3d,mask_cir,candi_mask] = circle_ROI (candi_mask,rad/2,cen_x,cen_y);
%                             length(find(overlay1~=0))
%                                 figure(1);cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','b','LineWidth',1.5);
%                         end
%                     else
%                         dis = [cen_y-col(1:i-1), cen_x-row(1:i-1)];
%                         dis_norm = zeros(size(dis,1),1);
%                         for k = 1:size(dis,1)
%                             dis_temp = dis(k,:);
%                             dis_norm(k) = norm(dis_temp);
%                         end
%                         if length( find ( dis_norm < rad ) )==0
%                             [mask_3d,mask_cir,candi_mask] = circle_ROI (candi_mask,rad,cen_x,cen_y);
%                             overlay1 = mask_anno.*mask_cir;
%                             overlay2 = lbls.*mask_cir; %indicate the overmap between current circle and lbls(lbls represents the annotation index)
%                             overlay0 = candi1.*mask_cir;
%                             if length(find(overlay1~=0))~=0
%                                 rad1 = sum(overlay1(:))/length(find(overlay1~=0))
%                                 if rad1~=0 && rad1>rad
%                                     nover = pi*rad*rad*ratio_th
%                                 elseif rad1~=0 && rad1<=rad
%                                     nover = pi*rad1*rad1*ratio_th
%                                 end
%                             else
%                                 nover = pi*rad*rad*ratio_th;
%                             end
%                             if length(find(mask_cir~=0))&& length(find(overlay1~=0))>nover && cen_y>rad-1 && cen_x>rad-1
%                                 num_pos = num_pos + 1;
%                                 num_pat_n = num_pat_n + 1;
%                                 yf1(num_pos) = rad1;
%                                                             im1 = uint8( double(im0).* mask_3d );
%                                                             pat1 = im1(cen_y-rad+1:cen_y+rad,cen_x-rad+1:cen_x+rad,:);
%                                                             pat_name = [path_pat1,sprintf('%s',num2str(num_pat_n),'.png')];
%                                                             imwrite(pat1,pat_name);
%                                 fprintf(file2,'%s %s %d %d %d %8.4f\n',foldername, filename_anno,num_pat,rad1);
%                                 figure(1); cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','r','LineWidth',1.5);
%                                 mask_anno(mask_cir~=0)=0;
%                                 mask_cmp = mask_cmp + mask_cir;
%                                 if num_pat_n==num_th
%                                     break;
%                                 end
%                             elseif length(find(mask_cir~=0))==length(find(overlay0~=0))&& cen_y>rad-1 && cen_x>rad-1
%                                 num_neg = num_neg + 1;
%                                 num_pat_n = num_pat_n + 1;
%                                                             im1 = uint8( double(im0).* mask_3d );
%                                                             pat1 = im1(cen_y-rad+1:cen_y+rad,cen_x-rad+1:cen_x+rad,:);
%                                                             pat_name = [path_pat0,sprintf('%s',num2str(num_pat_n),'.png')];
%                                                             imwrite(pat1,pat_name);
%                                 figure(1); cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','g','LineWidth',1);
%                                 if num_pat_n==num_th
%                                     break;
%                                 end
%                             else
%                                 candi_mask(mask_cir~=0) = 0;
%                                 [mask_3d,mask_cir,candi_mask] = circle_ROI (candi_mask,rad/2,cen_x,cen_y);
%                                 length(find(overlay1~=0))
%                                     figure(1);cent = [cen_x, cen_y]; viscircles(cent,rad,'Color','b','LineWidth',1.5);
%                                
%                             end
%                         end
%                         
%                     end
                end
                candi = candi.*(1-candi_mask);
                candi_temp = candi;
                candi_temp(candi_temp==0) = [];
                min_temp = min(candi_temp(:));
            end
            [m_try,pos,num_para,Nlab,num_over] = loadGT_new_withComp(file_anno,s1,mask_cmp(1:end-s1,1:end-s2));
            fprintf(file3,'%s %s %d %d %d %d\n',foldername, filename_anno,num_para,num_over, num_pos,num_neg );
        end
    end
end
fprintf(file2,'The number of total parasites are: '); fprintf(file2,'%d\n',num_pat);
fprintf(file2,'The mean radius of the all the parasites is: ');fprintf(file2,'%8.4f\n',mean(yf1(:)));
fclose(file2);
fclose(file3);
num_pat
mean(yf1(:))
%% randomly show 20 parasite figure
% figure;
% perm = randperm(num_pat,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(pats_pos(:,:,perm(i)),[])
% end
%%
disp(['Estimate the running time for preparing patches: ',num2str(etime(clock,t1))]);