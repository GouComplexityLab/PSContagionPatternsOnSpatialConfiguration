clear all
close all
clc

gwLineWidth = 1.0;
gwcolors = lines(6);
Netname = 'Delta12';
% Netname = 'Poisson12';
% Netname = 'Powerlaw12';

mat_name = ['./SeriesDataStep1RandomSeeding' Netname '.mat'];
load(mat_name, 'pvalues','ending_averagedI','ICns')

ICns = [1];

%%
% close all

figure
set(gcf,'Position', [100 300 600 360])
axes('Position',[0.1 0.2 0.80 0.75]);
set(gca,'Color','none')
hold on
for ICn = ICns
ICn
p=plot(pvalues,ending_averagedI(ICn,:),'MarkerSize',24,'Marker','.', ...
    'LineWidth',gwLineWidth,'LineStyle','--',...
    'Color','k');
% plot(pvalues,ending_averagedI,'MarkerSize',12,'Marker','.', ...
%     'LineWidth',gwLineWidth,'LineStyle','none',...
%     'Color',gwcolors(ICn,:));

% scatter(pvalues,ending_averagedI(ICn,:),30, pvalues,'filled');
% 
% colormap lines

end

ylim([-0.5 10.5])

ax = gca;
% xticks = xticks(ax);
xticks = [0.1,0.15,0.2,0.25,0.3];
xticklabels = arrayfun(@(x) sprintf('%.2f', x), xticks, 'UniformOutput', false);
xticklabels(find(xticks==0)) = {'0'};
% xticklabels(end) = {'$b$'};
set(gca, 'xtick', xticks);
set(gca, 'xticklabel', xticklabels);

% yticks = yticks(ax);
yticks = [0:5:10];
yticklabels = arrayfun(@(x) sprintf('%.0f', x), yticks, 'UniformOutput', false);
yticklabels(find(yticks==0)) = {'0'};
% yticklabels(end) = {'$a$'};
set(gca, 'ytick', yticks);
set(gca, 'yticklabel', yticklabels);

ax_FontSize = 24;
ax.XAxis.FontSize = ax_FontSize;  % 设置Y轴刻度标签的字体大小
ax.XAxis.FontName = 'Times New Roman';  % 设置Y轴刻度标签的字体类型
ax.XAxis.TickDirection = 'in';
ax.XAxis.TickLabelInterpreter = 'latex';


ax.YAxis.FontSize = ax_FontSize;  % 设置Y轴刻度标签的字体大小
ax.YAxis.FontName = 'Times New Roman';  % 设置Y轴刻度标签的字体类型
ax.YAxis.TickDirection = 'in';
ax.YAxis.TickLabelInterpreter = 'latex';

text(0.27,1.5,'$p$','FontName','Times New Roman','FontSize',24,'Interpreter','latex')

legend1 = legend([p],'$\langle I\rangle(T)$');
set(legend1,'Position',[0.110366030807504 0.785627457640048 0.23416690381864 0.117777776188321],...
    'Interpreter','latex',...
    'FontSize',ax_FontSize,'FontName','Informal Roman',...
    'EdgeColor','none','Color','none');

box on

% xlabel('Time $t$','FontName','Times New Roman','FontSize',24,'Interpreter','latex');
% ylabel('$I(t)$','FontName','Times New Roman','FontSize',24,'Interpreter','latex');

set(gca,'TickLength',...
    [0.01 0.02],'linewidth',gwLineWidth,'layer','top');

figure_name = ['./Step1 Random Seeding ' Netname '.eps'];
saveas(gcf, figure_name, 'epsc');









