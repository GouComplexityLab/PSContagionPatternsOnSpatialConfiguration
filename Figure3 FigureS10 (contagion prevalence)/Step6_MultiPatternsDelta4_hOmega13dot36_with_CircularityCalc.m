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
inmain_path = ['/New Step1 Cupy Delta4 Figure N=250000 Omega=' num2str(Omega) ' beta=0.01 nu=0.2'];
if Omega > 20
    inmain_path = ['/New Step1 Cupy Delta4 Figure N=250000 Omega=' num2str(Omega,'%.1f') ' beta=0.01 nu=0.2'];
end

main_path = ['./series data growth analysis'];
inmain_path = ['/New Step1 Cupy Delta4 Figure N=250000 Omega=' num2str(Omega) ' beta=0.01 nu=0.2'];
mat_name = ['/GrowthRateAnalysisDataIConnectivity3d_threshold=20.00.mat'];
load([main_path inmain_path mat_name])
%%
labeled_connectivityI_temp = zeros(size(labeled_connectivityI));

for eachz = 1:size(labeled_connectivityI,3)
    spliced_labeled_connectivityI = double(labeled_connectivityI(:,:,eachz));
    spliced_labeled_connectivityI = flipud(spliced_labeled_connectivityI);
%     spliced_labeled_connectivityI = rot90(spliced_labeled_connectivityI,-1);
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

    ns = [40 5 12];
    for eachn = 1:length(ns)
        n = ns(eachn);
        pos = find(spliced_labeled_connectivityI==C(n));

        if length(pos) > 0
            x_label_n = X2D(pos);
            y_label_n = Y2D(pos);
            x_label_n_one = x_label_n(1);
            y_label_n_one = y_label_n(1);

            underlying_grid = zeros(size(spliced_labeled_connectivityI));
            [InBoundarylobule_index, InnerPointslobule_index, OutBoundaryPointslobule_index] = findboundaries( underlying_grid,  pos, L);

            gwp = plot3(T*ones(size(InBoundarylobule_index)), X2D(InBoundarylobule_index), Y2D(InBoundarylobule_index), ...
                'Marker', '.', 'MarkerSize',6,'LineStyle','none', ...
                'Color',colormap_gw2(eachn,:));
            gwp.Color(4) = 0.3;

        end
    end

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

