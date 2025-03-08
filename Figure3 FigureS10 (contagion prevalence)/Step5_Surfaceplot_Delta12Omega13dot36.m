clear all
close all
clc

starting_time = datestr(now);

L = 500;

Omega=13.36;

main_path = ['./series data growth analysis'];
inmain_path = ['/New Step1 Cupy Delta12 Figure N=250000 Omega=' num2str(Omega) ' beta=0.01 nu=0.2'];
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

% labeled_connectivityI = permute(labeled_connectivityI_temp, [1, 3, 2]);

%%
% spliced_labeled_connectivityI = double(labeled_connectivityI(:,:,end));
spliced_labeled_connectivityI = double(labeled_connectivityI(:,:,100));

% unique(spliced_labeled_connectivity)



% figure
% set(gcf,'position',[100 300 1000 300])
% subplot(1,2,1)
% hold on
% surf(spliced_I_final);
% 
% % colorbar('FontSize',14);
% % set(gca,'xtick',[]);
% % set(gca,'ytick',[]);
% shading interp;
% colorbar;
% axis equal
% xlim([1 L])
% ylim([1 L])
% 
% % axis off
% view([0 90]);
% 
% subplot(1,2,2)
% hold on
% surf(spliced_labeled_connectivityI);
% colorbar;
% shading interp;
% axis equal
% xlim([1 L])
% ylim([1 L])
% % axis off
% view([0 90]);

figure
% set(gcf,'position',[100 200 600 600])
% set(gcf,'Position', [100 300 300 390])
set(gcf,'Position', [100 300 390 230])
hold on
% surf(spliced_labeled_connectivityI);
% colorbar;
% shading interp;
% axis equal
xlim([1 L])
ylim([1 L])
% axis off
view([0 90]);

%%

[Nx Ny Nz] = size(labeled_connectivityI);
x1d = [1:Nx];
y1d = [1:Ny];
z1d = [1:Nz];

[X2D, Y2D] = meshgrid(x1d, y1d);

[C] = unique(labeled_connectivityI(:));
C = double(C(2:end))';

% C = [1:108];

% colormap_gw = jet(length(C));
colormap_gw = jet(max(C));

% max_C = double(max(C));
max_C = 400;
number_C = zeros(size(C));
for i=1:length(C)
    number_C(i) = length(find(spliced_labeled_connectivityI==C(i)));
end

for label_n = C
    label_n
    pos = find(spliced_labeled_connectivityI==label_n);
    if length(pos) > 0
        x_label_n = X2D(pos);
        y_label_n = Y2D(pos);
        x_label_n_one = x_label_n(1);
        y_label_n_one = y_label_n(1);
        gwp = plot3(x_label_n, y_label_n, max_C*ones(size(pos)), ...
            'Marker', '.', 'MarkerSize',3,'LineStyle','none', ...
            'Color',colormap_gw(label_n,:));
        gwp.Color(4) = 0.3;
        text(x_label_n_one,y_label_n_one,max_C,[num2str(label_n) ',' num2str(length(pos))], ...
            'FontSize',9,'FontName','Times New Roman',...
            'Color','k','Visible','on');
    end
end
% set(gca,'xtick',[]);
% set(gca,'ytick',[]);
% axis equal
% axis off
% view([0 90]);

[X3D, Y3D, Z3D] = meshgrid(x1d, y1d, z1d);
%% plot label flat color
colormap_gw2 = [1 0.411764705882353 0.16078431372549;
                0.301960784313725 0.745098039215686 0.933333333333333;
                0 0 1];

figure
set(gcf,'Position', [100 300 660 360])
hold on

ns = [11 7 6];
for eachn = 1:length(ns)
    n = ns(eachn);
    disp(['n=' num2str(n)])
    I_final_n = labeled_connectivityI==C(n);
    s = isosurface(Z3D, X3D, Y3D, I_final_n, 0.9);
    p = patch(s);

    set(p,'FaceColor',colormap_gw2(eachn,:));  
    set(p,'EdgeColor','none');
    set(p,'FaceAlpha',0.9)
    set(p,'DisplayName', [num2str(n) ',' num2str(C(n))])
end

% camlight; lighting gouraud;

% title(['time t=' num2str(t,'%.1f')])
xlim([0 400+1])
ylim([0 L+1])
zlim([0 L+1])

% ylim([0 Nz+1])

% ylim([0 Nz+1])

% set(gca,'xtick',[500,1000:1000:4000]/10);
% set(gca,'xticklabels',[500,1000:1000:4000]);
set(gca,'ytick',[100,250,400]);
set(gca,'yticklabels',[]);
set(gca,'ztick',[100,250,400]);
set(gca,'zticklabels',[]);

ax1 = gca;
xticks = [500,1000:1000:4000]/10;
xticklabels = arrayfun(@(x) sprintf('%d', x), [500,1000:1000:4000], 'UniformOutput', false);
% xticklabels(1) = {'0'};
set(gca, 'xtick', xticks);
set(gca, 'xticklabel', xticklabels);

ax1_FontSize = 18;
ax1.XAxis.FontSize = ax1_FontSize;  % 设置Y轴刻度标签的字体大小
ax1.XAxis.FontName = 'Times New Roman';  % 设置Y轴刻度标签的字体类型
ax1.YAxis.FontSize = ax1_FontSize;  % 设置Y轴刻度标签的字体大小
ax1.YAxis.FontName = 'Times New Roman';  % 设置Y轴刻度标签的字体类型
ax1.XAxis.TickDirection = 'in';
ax1.YAxis.TickDirection = 'in';
ax1.ZAxis.TickDirection = 'in';
ax1.XAxis.TickLabelInterpreter = 'latex';
ax1.YAxis.TickLabelInterpreter = 'latex';
% xlabel('X');
% ylabel('Y');
% zlabel('z');

% view([-5 25]);
view([12 25]);


grid on;
set( gca, 'Box', 'on', 'BoxStyle', 'full' );

% camorbit(90,0,'data',[0,1,0])%[0 0 1]表示按z轴旋转。36*10=360表示旋转一周
% ax = gca;  % 获取当前坐标轴对象
% ax.ZAxisLocation = 'right';
% set(gca,'ZAxisLocation','right');

end_time = datestr(now);
disp(['starting_time =' starting_time ', end_time = ' end_time])





