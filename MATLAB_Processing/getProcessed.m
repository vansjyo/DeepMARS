%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) Reads : *foc.png & *1024-Sim.png
% (2) Convert images to grayscale and resize
% (3) Register the simulated image
% (4) Creates register.tif, resize.tif
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
a=dir('*.foc.png');
b=dir('*1024-Sim.png');

for i=1:size(a,1)
    
    track = a(i).name;
    foc = imread(a(i).name);
    sim = imread(b(i).name);
    
    foc_resize = imresize(foc, [1024 size(foc,2)]);
    sim_mat = rgb2gray(sim);
    
    moving = sim_mat;
    fixed = foc_resize;

    %     [optimizer,metric] = imregconfig('multimodal');
    %     optimizer.InitialRadius = optimizer.InitialRadius/3.5;
    %     optimizer.MaximumIterations = 300;
    %     tformSimilarity = imregtform(moving,fixed,'similarity',optimizer,metric);
    %     movingRegisteredAffineWithIC = imregister(moving,fixed,'similarity',optimizer,metric,'InitialTransformation',tformSimilarity);

    % Register simulated
    tformEstimate = imregcorr(moving,fixed);
    Rfixed = imref2d(size(fixed));
    movingReg = imwarp(moving,tformEstimate,'OutputView',Rfixed);
    
    % fetch name
    if track(14)=='_'
        name = track(5:27);
    else
        name = track(5:28);
    end
    
    % write to file
    imwrite(movingReg, strcat('register_',name,'.tif'));
    imwrite(foc_resize, strcat('resize_',name,'.tif'));
    
    % create norm file
    %norm = mat2gray(foc_resize) - mat2gray(movingReg);
    %norm = (norm - min(min(norm)))./(max(max(norm)) - min(min(norm)));
    %imwrite(double(norm),strcat('norm_',name,'.tif'));

    % can ommit this part; only to reduce rows
    %     imwrite(movingReg(425:680,:),'F_register.tif');
    %     imwrite(foc_resize(425:680,:),'F_resize.tif');
    %     imwrite(norm(425:680,:),'F_norm.tif');
    % 
    %     b1 = imread('F_resize.tif');
    %     b2 = imread('F_register.tif');
    %     b3 = imread('F_norm.tif');
    % 
    %     band3(:,:,1) = b1;
    %     band3(:,:,2) = b2;
    %     band3(:,:,3) = b3;
    %     imwrite(band3,'F_220201000_3bands.tif');
end