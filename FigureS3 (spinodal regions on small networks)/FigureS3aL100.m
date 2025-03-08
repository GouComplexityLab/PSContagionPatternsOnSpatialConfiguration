clear all
close all

clc

beta = 0.01; nu = 0.2; gamma = 1 / 7;

%%
fig = figure;
set(gcf,'Position', [100 300 560 420])
ax1 = axes('Position',[0.0919642857142857 0.154761904761905 0.85267857142857 0.773809523809524]);

set(gca,'Color','none')

gwLineWidth = 1.5;

left_color = [0 0 1];
right_color = [0 0 0];
set(fig,'defaultAxesColorOrder',[left_color; right_color]);

hold on


%%
netname = 'NetDelta12L100';
% netname = 'NetPoisson12L100';
% netname = 'NetPowerlaw12L100';
data_name = [netname 'RegionDataSmall_beta=' dot2d(beta) '_nu=' dot2d(nu) '.mat'];
eval(['load ' data_name])

load Delta12eigenvalues_with_L=100 first_smallest_eigenvalues
Lambda2_Delta12 = first_smallest_eigenvalues(2);

x1 = N_values(find(isnan(critical_d_s_min1_values1)==0)); 
y1 = critical_d_s_min1_values1(find(isnan(critical_d_s_min1_values1)==0));
x2 = N_values(find(isnan(critical_d_s_max1_values1)==0));
y2 = critical_d_s_max1_values1(find(isnan(critical_d_s_max1_values1)==0));

f1 = plot(x1,y1,'Color', 'b','LineWidth',gwLineWidth,'LineStyle','-');
plot([x1(1),x2(1)],[y1(1),y2(1)],'Color', 'b','LineWidth',gwLineWidth,'LineStyle','-');


%%
% netname = 'NetDelta12L100';
netname = 'NetPoisson12L100';
% netname = 'NetPowerlaw12L100';
data_name = [netname 'RegionDataSmall_beta=' dot2d(beta) '_nu=' dot2d(nu) '.mat'];
eval(['load ' data_name])

x1 = N_values(find(isnan(critical_d_s_min1_values1)==0));
y1 = critical_d_s_min1_values1(find(isnan(critical_d_s_min1_values1)==0));
x2 = N_values(find(isnan(critical_d_s_max1_values1)==0));
y2 = critical_d_s_max1_values1(find(isnan(critical_d_s_max1_values1)==0));

f2 = plot(x1,y1,'Color', 'b','LineWidth',gwLineWidth,'LineStyle','--');
plot([x1(1),x2(1)],[y1(1),y2(1)],'Color', 'b','LineWidth',gwLineWidth,'LineStyle','--');


%%
% netname = 'NetDelta12L100';
% netname = 'NetPoisson12L100';
netname = 'NetPowerlaw12L100';
data_name = [netname 'RegionDataSmall_beta=' dot2d(beta) '_nu=' dot2d(nu) '.mat'];
eval(['load ' data_name])

x1 = N_values(find(isnan(critical_d_s_min1_values1)==0));
y1 = critical_d_s_min1_values1(find(isnan(critical_d_s_min1_values1)==0));
x2 = N_values(find(isnan(critical_d_s_max1_values1)==0));
y2 = critical_d_s_max1_values1(find(isnan(critical_d_s_max1_values1)==0));

ax = gca; 
ax.XTick = [[0:10:30],38]; 
ax.XTickLabel = {'0', '10', '20', '30', '$\langle N\rangle$'}; 
ax.XAxis.FontName = 'Times New Roman';
ax.TickLabelInterpreter = 'latex';

ax.YTick = [1,[4:4:16],20]; 
ax.YTickLabel = {'1', '4', '8', '12', '16', '$d_{S}$'}; 
ax.YAxis.FontName = 'Times New Roman'; 
ax.TickLabelInterpreter = 'latex';

ax.XAxis.TickDirection = 'in';
ax.YAxis.TickDirection = 'in';

box on
N_max = 40;
xlim([0 N_max])
ylim([0 20])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set(gca,'XColor','k','YColor','b','TickLength',...
    [0.02 0.02],'FontSize',24,'linewidth',gwLineWidth,'layer','top');

%%
netname = 'NetDelta12LX';
data_name = [netname 'RegionDataSmall_beta=' dot2d(beta) '_nu=' dot2d(nu) '.mat'];
eval(['load ' data_name])

load Delta12eigenvalues_with_L=500 first_smallest_eigenvalues
Lambda2_Delta12 = first_smallest_eigenvalues(2);

x1 = N_values(find(isnan(critical_d_s_min1_values1)==0)); 
y1 = critical_d_s_min1_values1(find(isnan(critical_d_s_min1_values1)==0));
x2 = N_values(find(isnan(critical_d_s_max1_values1)==0));
y2 = critical_d_s_max1_values1(find(isnan(critical_d_s_max1_values1)==0));

f4 = plot(x1,y1,'Color', 'k','LineWidth',gwLineWidth,'LineStyle','-');
plot([x1(1),x2(1)],[y1(1),y2(1)],'Color', 'k','LineWidth',gwLineWidth,'LineStyle','-');
%%

text(18,16,{'spinodal', 'region'},'FontSize',18,'FontName','Times New Roman',...
    'HorizontalAlignment','center','Interpreter','latex');

annotation('arrow',[0.35 0.35+0.05],...
    [0.773809523809527 0.773809523809527],'LineWidth',2);
annotation('arrow',[0.56+0.13 0.56],...
    [0.773809523809527 0.773809523809527],'LineWidth',2);

load Delta12eigenvalues_with_L=100 first_smallest_eigenvalues
Lambda2_Delta12 = first_smallest_eigenvalues(2);

load poisson12eigenvalues_with_L=100 first_smallest_eigenvalues
Lambda2_poisson12 = first_smallest_eigenvalues(2);

load powerlaw12eigenvalues_with_L=100 first_smallest_eigenvalues
Lambda2_powerlaw12 = first_smallest_eigenvalues(2);

annotation('textbox', [0.70 0.25+0.07*3 0.18 0.06],...
           'String', {['$\Lambda_{2}=' num2str(Lambda2_Delta12,'%.3f') '$']}, ...  
           'Color', 'k', 'FontSize', 12, 'EdgeColor', 'b','LineStyle','-','LineWidth',1.2,'Interpreter','latex'); 
annotation('textbox', [0.70 0.25+0.07*2 0.18 0.06],...
           'String', {['$\Lambda_{2}=' num2str(Lambda2_poisson12,'%.3f') '$']}, ...  
           'Color', 'k', 'FontSize', 12, 'EdgeColor', 'b','LineStyle','--','LineWidth',1.2,'Interpreter','latex'); 
annotation('textbox', [0.70 0.25+0.07*1 0.18 0.06],...
           'String', {['$\Lambda_{2}=' num2str(Lambda2_powerlaw12,'%.3f') '$']}, ...  
           'Color', 'k', 'FontSize', 12, 'EdgeColor', 'b','LineStyle',':','LineWidth',1.2,'Interpreter','latex'); 

annotation('textbox', [0.70 0.25 0.18 0.06],...
           'String', {['$\Lambda_{2}\longmapsto' num2str(0) '$']}, ...  
           'Color', 'k', 'FontSize', 12, 'EdgeColor', 'k','LineStyle','-','LineWidth',1.2,'Interpreter','latex');  

figure_name = ['./FigureS3aL100.eps'];
saveas(gcf, figure_name, 'epsc');

