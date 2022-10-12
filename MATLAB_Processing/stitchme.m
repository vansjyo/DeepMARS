%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) Reads : finalfigure*.tif 256x256 image
% (2) Creates stitched original size image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
files =  dir('finalfigure_test_999*.tif');
f_size = size(files,1);
f = zeros(256,256,f_size);
for i=1:size(files,1)
    f(:,:,i) = imread(files(i).name);
end
y_limit = 1;
x_limit = 5;
stitch = zeros(256*y_limit, 256*x_limit);
count = 1;
for i=1:y_limit
    
    y_start = 256*(i-1)+1;
    y_finish = 256*i;
    
    for j=1:x_limit
        
        x_start = 256*(j-1)+1;
        x_finish = 256*j;
   
        stitch(y_start:y_finish, x_start:x_finish) = f(:,:,count);
        count = count +1;
        
    end
    
end
track = files(1).name;
name = "stitched_" + track(1:(end-10)) + ".tif";
imwrite(stitch, name);