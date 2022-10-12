mola_file = dir('*4096-Sim.png');
for k=1:size(mola_file,1)
    mola = imread(mola_file(k).name);
    [rows, cols, ~] = size(mola);
    topo = zeros(rows,cols);

    for i=1:cols
        for j=1:rows
            if(mola(j,i,1)==255 && mola(j,i,2)==255 && mola(j,i,3)==0) 
                topo(j,i) = 1;
            end
        end
    end
    
    file = mola_file(k).name;
    name = file(5:14);
    file_name = strcat('topo_', name,'.tif');
    imwrite(topo,file_name);
    
    t = imread(file_name);
    t = imresize(t, [rows/4 cols]);
    file_name = strcat('topo_resize_', name,'.tif');
    imwrite(t,file_name);
end

data_file = dir('QDA_*latlon.txt');
label_file = dir('Label_*.tif');
topo_file = dir('topo_resize*.tif');

for k=1:size(data_file,1)
    clear rows cols;
    data = importdata(data_file(k).name);
    label = imread(label_file(k).name);
    topo = imread(topo_file(k).name);
    
    %     stitch = zeros(size(topo,1), size(topo,2));
    %     stitch(425:680,:) = imbinarize(label(:,1:size(topo,2)));
    %     clear label;
    %     label = stitch;
    
    [rows,cols] = find(label);
    info = zeros( size(rows,1), 6);
    dT = 37.5*10^(-9);
    c = 3*10^8;
    v = (3*10^8)/(3.15^0.5);
    dD = (dT/2)*c;
    dD1 = (dT/2)*v;

    for i=1:size(rows,1)
        mod_cols = cols(i)*8;
        %         if(k==6)
        %             mod_cols = (cols(i)+656)*8;
        %         end
        a = find(topo(:,cols(i)));
        mod_rows = rows(i);
        if(mod_rows - a(1))
            depth = a(1)*dD + (mod_rows - a(1))*dD1 - a(1)*dD;
        else
            depth = mod_rows*dD - a(1)*dD;
        end
        info(i,:) = [i rows(i) cols(i) data(mod_cols,3) data(mod_cols,4) depth];
    end
    
    file = data_file(k).name;
    name = file(5:14);
    %     if(k==6)
    %         name = file(5:18);
    %     end
    file_name = strcat('INFO_Stitched_', name, '.csv');
    T = array2table(info);
    T.Properties.VariableNames(1:6) = {'SNo','Row','Column','Latitude','Longitude','Depth'};
%     csvwrite(file_name, info);
    writetable(T, file_name);
    dlmwrite('info.csv',info,'-append');
   
end
