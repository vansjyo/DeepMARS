%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) Reads : register.tif & resize.tif &  Label.tif
% (2) Creates norm feature
% (3) Crops image and create F_* images
% (4) Creates 3 band image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
a = dir('resize*.tif');
b = dir('register*.tif');
% c = dir('Label*.tif');

for i=1:size(a,1)
    
    clear band3;
    
    foc = imread(a(i).name);
    sim = imread(b(i).name);
%     label = imread(c(i).name);
    
    moving = mat2gray(sim);
    fixed = mat2gray(foc);
    
    % fetch name - set index accordingly
    track = a(i).name;
    if track(17)=='.'
        name = track(8:16);
    else
        name = track(8:15);
    end    
   
    % create norm file
    norm = mat2gray(foc) - mat2gray(moving);
    norm = (norm - min(min(norm)))./(max(max(norm)) - min(min(norm)));
    imwrite(norm,strcat('norm_',name,'.tif'));

    % can ommit this part; only to reduce rows
    %     crop = 144:399;
    imwrite(moving(:,:),strcat('F_register_',name,'.tif'));
    imwrite(foc(:,:),strcat('F_resize_',name,'.tif'));
    imwrite(norm(:,:),strcat('F_norm_',name,'.tif'));

    b1 = imread(strcat('F_resize_',name,'.tif'));
    b2 = imread(strcat('F_register_',name,'.tif'));
    b3 = imread(strcat('F_norm_',name,'.tif'));

    band3(:,:,1) = b1;
    band3(:,:,2) = b2;
    band3(:,:,3) = b3;
    imwrite(band3,strcat('F_3bands_',name,'.tif'));
    
end