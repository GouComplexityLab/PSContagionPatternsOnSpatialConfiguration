clear all
close all
clc

load('./gwcustom_cmap.mat')
gwcustom_cmap = gwcolormap(:,1:3);

starting_time = datestr(now);

L = 500;
Omega=13.36;

Ts = [200:200:1000];
Is_reshaped = zeros(L,L,length(Ts));
Ss_reshaped = zeros(L,L,length(Ts));

net_name = 'Delta12';
p = 0.180;
p = 0.190;
p = 0.300;

data_path = ['./PreparedData Step1 Cupy ' net_name ' Figure ICn=1 N=250000 Omega=13.36 beta=0.01 nu=0.2 p=' num2str(p,'%.3f') ];


%%
for Tn=1:length(Ts)
    T = Ts(Tn);
    mat_name = ['/SpatialConfigurationPattern T=' num2str(T,'%.4f') '.mat'];
    load([data_path mat_name])
    
    I_reshaped = reshape(I0,L,L);
    S_reshaped = reshape(S0,L,L);
    
    % I_reshaped = flipud(I_reshaped);
    I_reshaped = rot90(I_reshaped,1);
    
    % S_reshaped = flipud(S_reshaped);
    S_reshaped = rot90(S_reshaped,1);
    
    Is_reshaped(:,:,Tn) = I_reshaped;
    Ss_reshaped(:,:,Tn) = S_reshaped;

end

% 创建一个新的图形窗口
figure
set(gcf,'Position', [100 300 660 360])
% axes('Position',[0.15 0.20 0.798 0.725]);
set(gca,'Color','none')

hold on;

% view([-5 25]);
view([12 25]);

% 使用透明度和渐变效果绘制每一层图像
alpha_val = 0.95; % 透明度

[X, Y] = meshgrid(1:L, 1:L);
Ts_ = Ts;
for Tn=1:length(Ts)
    T = Ts(Tn);

    surf(T * ones(size(X)), X, Y, Is_reshaped(:,:,Tn), 'EdgeColor', 'none');
    alpha(alpha_val);

    shading interp;

    colormap_gw2 = [1 0.411764705882353 0.16078431372549;
                    0.301960784313725 0.745098039215686 0.933333333333333;
                    0 0 1];

end


colormap(gwcustom_cmap); % 选择适当的颜色映射
% colorbar; % 显示颜色条
caxis([0 45])


set(gca,'xtick',[]);
set(gca,'ytick',[]);
set(gca,'ztick',[]);

% set(gca,'ZColor','none');
% set(gca,'YColor','none');
% set(gca,'XColor','none');

xlim([1 L])
ylim([1 L])
zlim([0 1000+1])
axis tight;
axis off

% % annotation('line',[0.86 0.86],...
% %     [0.08 0.98]);
pianx = 0.17;
annotation('textbox',...
    [0.15 0.10 0.15 0.08],...
    'Rotation',45,...
    'String',{'$t=200$'},...
    'Interpreter','latex',...
    'FontSize',18,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');

annotation('textbox',...
    [0.15+pianx 0.10 0.15 0.08],...
    'Rotation',45,...
    'String',{'$t=400$'},...
    'Interpreter','latex',...
    'FontSize',18,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');

annotation('textbox',...
    [0.15+pianx*2-0.005 0.10 0.15 0.08],...
    'Rotation',45,...
    'String',{'$t=600$'},...
    'Interpreter','latex',...
    'FontSize',18,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');

annotation('textbox',...
    [0.15+pianx*3-0.005*2 0.10 0.15 0.08],...
    'Rotation',45,...
    'String',{'$t=800$'},...
    'Interpreter','latex',...
    'FontSize',18,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');


annotation('textbox',...
    [0.15+pianx*4-0.005*3 0.10 0.15 0.08],...
    'Rotation',45,...
    'String',{'$t=1000$'},...
    'Interpreter','latex',...
    'FontSize',18,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');


