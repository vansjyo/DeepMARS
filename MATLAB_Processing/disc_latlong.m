data = dir('QDA_*latlon.txt');
data = importdata(data(1).name);
k=1;
%b = a1(:,:,k);
b = dir('bound_*F.tif');
b = importdata(b(1).name);
%b=imread('bound_7832_F.tif');
%b = imbinarize(b,0.15);
% b = imread('bound_5795_F.tif');
m = imread('topo_resize.tif');

if(k==1 || k==2)
    y1 = [1072 1327]; %anonymous LDA
end
if(k==3 || k==4)
    y1 = [1404 1659]; %Euripus mons
end

if(k==1 || k==3)
    x2 = [1 256]; %after cut top
end
if(k==2 || k==4)
    x2 = [176 431]; %after cut botton
end
[rows,cols] = find(b);
info = zeros( size(rows,1), 6);
dT = 37.5*10^(-9);
c = 3*10^8;
v = (3*10^8)/(3.15^0.5);
dD = (dT/2)*c;
dD1 = (dT/2)*v;
y1=[1 1];

x1 =[1 1]; %initial cut for image
x2=[1 1];
%[index row col lat lo  ng depth]
for i=1:size(rows,1)
    mod_cols = (y1(1)-1+cols(i))*8;
    a = find(m(:,y1(1)-1+cols(i)));
    mod_rows = x1(1)-1+x2(1)-1+rows(i);
    if(mod_rows - a(1))
        depth = a(1)*dD + (mod_rows - a(1))*dD1 - a(1)*dD;
    else
        depth = mod_rows*dD - a(1)*dD;
    end
    info(i,:) = [i rows(i) cols(i) data(mod_cols,3) data(mod_cols,4) depth];
end
m=256;
file_name = strcat('INFO_1_1_',num2str(m),'.csv');
writematrix(info,file_name);
dlmwrite('D:\VANSHIKA\info.csv',info,'-append');
