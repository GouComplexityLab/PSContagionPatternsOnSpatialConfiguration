clear all
close all
clc

load('./gwcustom_cmap.mat')
gwcustom_cmap = gwcolormap(:,1:3);

starting_time = datestr(now);

L = 500;
Omega=13.36;
% Omega=22.0;
% Omega=26.0;

Ts = [200,500,1000];
Ts = [500,1000:1000:4000];
Is_reshaped = zeros(L,L,length(Ts));
Ss_reshaped = zeros(L,L,length(Ts));

T = 700;

main_path = ['./series data growth analysis'];
inmain_path = ['/New Step1 Cupy Poisson12 Figure N=250000 Omega=' num2str(Omega) ' beta=0.01 nu=0.2'];
if Omega > 20
    inmain_path = ['/New Step1 Cupy Poisson12 Figure N=250000 Omega=' num2str(Omega,'%.1f') ' beta=0.01 nu=0.2'];
end

main_path = ['./series data growth analysis'];
inmain_path = ['/New Step1 Cupy Poisson12 Figure N=250000 Omega=' num2str(Omega) ' beta=0.01 nu=0.2'];
mat_name = ['/GrowthRateAnalysisDataIConnectivity3d_threshold=20.00.mat'];
load([main_path inmain_path mat_name])
%%
labeled_connectivityI_temp = zeros(size(labeled_connectivityI));

for eachz = 1:size(labeled_connectivityI,3)
    spliced_labeled_connectivityI = double(labeled_connectivityI(:,:,eachz));
    spliced_labeled_connectivityI = flipud(spliced_labeled_connectivityI);
    labeled_connectivityI_temp(:,:,eachz) = spliced_labeled_connectivityI;
end

labeled_connectivityI = labeled_connectivityI_temp;

[C] = unique(labeled_connectivityI(:));
C = double(C(2:end))';

[Nx Ny Nz] = size(labeled_connectivityI);
x1d = [1:Nx];
y1d = [1:Ny];
z1d = [1:Nz];

[X2D, Y2D] = meshgrid(x1d, y1d);

%%
T= 4000;
mat_name = ['/SpatialConfigurationPattern T=' num2str(T,'%.4f') '.mat'];
load([main_path inmain_path mat_name])

I_reshaped = reshape(I0,L,L);
S_reshaped = reshape(S0,L,L);

% I_reshaped = flipud(I_reshaped);
I_reshaped = rot90(I_reshaped,1);

figure
D = 200;
set(gcf,"Position",[80 360 D D])
hold on
s = surf(I_reshaped);
s.LineStyle = 'none';

axis equal 
xlim([1, size(I_reshaped,2)])
ylim([1, size(I_reshaped,1)])



extended_I = repmat(I_reshaped, 3, 3);

figure
hold on
set(gcf,"Position",[500 90 D*3 D*3])
hold on
s = surf(extended_I);
s.LineStyle = 'none';

x = [L, L*2, L*2, L, L];
y = [L, L, L*2, L*2, L];
z = [50, 50, 50, 50, 50];
plot3(x, y, z, 'k', 'LineWidth', 2);
xlim([1, size(extended_I,2)])
ylim([1, size(extended_I,1)])

%%
mask_I = extended_I>20;
[label_I,Num] = bwlabel(mask_I,4);

figure
hold on
set(gcf,"Position",[500 90 D*3 D*3])
hold on
s = surf(label_I);
s.LineStyle = 'none';

x = [L, L*2, L*2, L, L];
y = [L, L, L*2, L*2, L];
z = [50, 50, 50, 50, 50];
plot3(x, y, z, 'k', 'LineWidth', 2);

label_n = 56;

many_centroids = [];
many_Circularity = [];
for label_n = 1:Num

    pos_index = find(label_I == label_n);
    [pos_x, pos_y] = ind2sub([L*3, L*3], pos_index);
    
    [in, on] = inpolygon(pos_y, pos_x, x, y);
    pos_in = find(in==1);
    if length(find(in==1))>0 && length(pos_y)>1000
        disp(['label_n = ' num2str(label_n) ' is in simulation region.'])
    
        mask_n = label_I == label_n;
        
        s = regionprops(mask_n,'centroid');
        centroids = cat(1,s.Centroid);
        plot3(centroids(1), centroids(2), 180, 'k.', 'LineStyle', 'none');
        
        Obj = regionprops(mask_n,'Circularity');
        Circularity = cat(1,Obj.Circularity); 
        text(centroids(1), centroids(2), 180, num2str(Circularity,'%.2f'), ...
            'FontSize',18,'FontName','Times New Roman',...
            'Interpreter','latex','Color','k'); 
        many_centroids = [many_centroids;centroids];
        many_Circularity = [many_Circularity;Circularity];
    end

end

xlim([L, L*2])
ylim([L, L*2])



%%
for Tn=1:length(Ts)
    T = Ts(Tn);
    mat_name = ['/SpatialConfigurationPattern T=' num2str(T,'%.4f') '.mat'];
    load([main_path inmain_path mat_name])
    
    I_reshaped = reshape(I0,L,L);
    S_reshaped = reshape(S0,L,L);
    
    % I_reshaped = flipud(I_reshaped);
    I_reshaped = rot90(I_reshaped,1);
    
    % S_reshaped = flipud(S_reshaped);
    S_reshaped = rot90(S_reshaped,1);
    
    Is_reshaped(:,:,Tn) = I_reshaped;
    Ss_reshaped(:,:,Tn) = S_reshaped;

end


figure
set(gcf,'Position', [100 300 660 360])
set(gca,'Color','none')

hold on;

view([12 25]);

alpha_val = 0.95;

[X, Y] = meshgrid(1:L, 1:L);
Ts = [500:1000:3500, 4500];
Ts_ = [500:1000:3500, 4000];
for Tn=1:length(Ts)
    T = Ts(Tn);

    surf(T * ones(size(X)), X, Y, Is_reshaped(:,:,Tn), 'EdgeColor', 'none');
    alpha(alpha_val);

    shading interp;

    colormap_gw2 = [1 0.411764705882353 0.16078431372549;
                    0.301960784313725 0.745098039215686 0.933333333333333;
                    0 0 1];

    spliced_labeled_connectivityI = double(labeled_connectivityI(:,:,Ts_(Tn)/10));

end

for each = 1:length(many_Circularity)
    centroids = many_centroids(each,:);
    Circularity = many_Circularity(each);
    text(4590, mod(centroids(1),L), mod(centroids(2),L), num2str(Circularity,'%.2f'), ...
        'Rotation',68,...
        'FontSize',12,'FontName','Times New Roman',...
        'Interpreter', 'latex', 'Color', 'k', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle');
end


colormap(gwcustom_cmap); 
caxis([0 45])


set(gca,'xtick',[]);
set(gca,'ytick',[]);
set(gca,'ztick',[]);


xlim([1 L])
ylim([1 L])
zlim([0 4000+1])
axis tight;
axis off

pianx = 0.17;
annotation('textbox',...
    [0.15 0.10 0.15 0.08],...
    'Rotation',45,...
    'String',{'$t=500$'},...
    'Interpreter','latex',...
    'FontSize',18,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');

annotation('textbox',...
    [0.15+pianx 0.10 0.15 0.08],...
    'Rotation',45,...
    'String',{'$t=1000$'},...
    'Interpreter','latex',...
    'FontSize',18,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');

annotation('textbox',...
    [0.15+pianx*2-0.005 0.10 0.15 0.08],...
    'Rotation',45,...
    'String',{'$t=2000$'},...
    'Interpreter','latex',...
    'FontSize',18,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');

annotation('textbox',...
    [0.15+pianx*3-0.005*2 0.10 0.15 0.08],...
    'Rotation',45,...
    'String',{'$t=3000$'},...
    'Interpreter','latex',...
    'FontSize',18,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');


annotation('textbox',...
    [0.15+pianx*4-0.005*3 0.10 0.15 0.08],...
    'Rotation',45,...
    'String',{'$t=4000$'},...
    'Interpreter','latex',...
    'FontSize',18,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');

