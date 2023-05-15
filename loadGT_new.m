% the difference between loadGT_new.m and loadGT_new_1.m is that the former
% does not return the para_position.
 function [GT,lbls,mask,parasitemic,Nlab] = loadGT_new(datfile,s1)

%%% rad can be changed according to the s1 and s2
% Open data file with same name as image file
% datfile = strcat(imfile(1:length(imfile)-4),'.txt');% for mouse
% datfile = strcat(imfile,'_','.txt');% for human
%  datfile = file_anno;
%  figure;imshow(im0);s1=44;
fid = fopen(datfile);
parasitemic = 0;
num_wbc = 0;
rad0 = 0.5*s1; %% rad is 1/2 of the s1 or s2;
if fid ~= -1
    s = textscan(fgetl(fid),'%s','delimiter',',');
    s=s{1}; 
    %%added by feng 11/6/2017
    Nlab = str2num(s{1})
    if Nlab ==0
        disp('Annotation is not available. Please choose another image');
        return
    end
    %%end by feng 11/6/2017
    mask = zeros(str2num(s{3}),str2num(s{2}));
    mask_points = zeros(str2num(s{3}),str2num(s{2}));
    GT = zeros(str2num(s{3}),str2num(s{2}));
    lbls = zeros(str2num(s{3}),str2num(s{2}));
    endh = 1;
    endauto = 1;
    endoffile = false;
    % For each line in the file
    while ~endoffile %~feof(fid)
        line = fgetl(fid); % Get next line %% changed by feng 11/28/2017
        row = textscan(line,'%s','delimiter',',');
        row = row{1};
        %Plot points
        if strcmp(row{4}, 'Circle')
            % the ROI is a circle, but I would like draw a rectangle here:
            cx =  round(sscanf(row{6},'%f')) ; %% the center of the circle or rectangle
            cy =  round(sscanf(row{7},'%f')) ;
            r1 = ( -round(sscanf(row{6},'%f')) + round(sscanf(row{8},'%f')) ) ;
            r2 = ( -round(sscanf(row{7},'%f')) + round(sscanf(row{9},'%f')) ) ;
            rad = round(sqrt( r1^2 + r2^2 ));
%             cent = [cx, cy];
%             figure(1); hold on;viscircles(cent,rad,'LineWidth',1);           
            r = [ cx-rad+1; cx-rad+1; cx+rad; cx+rad]; 
            c = [ cy-rad+1; cy+rad; cy+rad; cy-rad+1];
            mask_rect = roipoly(mask,r,c);
            mask = mask + mask_rect;
            if strcmp(row{2},'Parasite') 
                parasitemic = parasitemic+1;
                lbls(mask_rect) = 2;
                lbls(mask_rect) = parasitemic+10;
                para_pos(parasitemic,:) = [cx-rad+1 cy-rad+1 rad*2 rad*2];
                %pats_pos(1:s1,1:s2,parasitemic)= mask_rect.* im./2;
            else
                lbls(mask_rect)=1;
            end
%             yf(parasitemic)=rad1;
%             mean(yf(:))
        %%end by feng 11/06/2017
        elseif strcmp(row{4}, 'Polygon')
            disp('Attention: Polygon is used for parasite annotation, please check!');
            break;
        elseif strcmp(row{4}, 'Point')
            x = round(sscanf(row{6},'%f'));
            y = round(sscanf(row{7},'%f'));
            if x==0 x=1; end
            if y==0 y=1; end
            %mask_points(y,x)= 255; %% we can replace it with: 
            mask_points(y-50+1:y+50,x-50+1:x+50)= 1;
            if strcmp(row{2},'Parasite')
                parasitemic = parasitemic+1;
                %lbls(y,x) = 2;
                lbls(y,x) = parasitemic+10;%% to distinguish with white blood cells which are noted by 1
            else
                lbls(y,x)=1;
                num_wbc = num_wbc + 1;
            end
        end
        if feof(fid)
            endoffile = true;
        end 
        %line = fgetl(fid); % Get next line
    end
    fclose(fid); 
   GT = mask + mask_points;
   if parasitemic == 0
       para_pos = 0;
   end
%    GT(GT>=2)=1;
%    imshow(GT,[])

end
end