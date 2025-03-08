clear all
close all

clc

netname = 'NetPowerlaw12L500';

pre_figname = netname;
beta = 0.01; nu = 0.2/4; gamma = 1 / 7; 
dI = 1;
dS_critical = beta*dI/gamma/nu;
disp(['dS must be larger than: ' num2str(dS_critical)])

data_name = [netname 'RegionDataSmall_beta=' dot2d(beta) '_nu=' dot2d(nu) '.mat'];
eval(['load ' data_name])

fig = figure;
set(gcf,'Position', [100 300 560 420])
axes('Position',[0.0919642857142857 0.154761904761905 0.85267857142857 0.773809523809524]);
set(gca,'Color','none')

gwLineWidth = 1.5;

left_color = [0 0 1];
right_color = [0 0 0];
set(fig,'defaultAxesColorOrder',[left_color; right_color]);

% yyaxis left
hold on
x1 = N_values(find(isnan(critical_d_s_min1_values1)==0)); 
y1 = critical_d_s_min1_values1(find(isnan(critical_d_s_min1_values1)==0));

x2 = N_values(find(isnan(critical_d_s_max1_values1)==0));
y2 = critical_d_s_max1_values1(find(isnan(critical_d_s_max1_values1)==0));


left_mostmin_N = min(x1);
left_mostmin_ds = min(y1);

left_mostmax_N = min(x2);
left_mostmax_ds = min(y2);

x_fill = [x1, fliplr(x2)];
y_fill = [y1, fliplr(y2)];
f1 = fill(x_fill, y_fill,'b', 'FaceAlpha', 0.3,'EdgeColor', 'b','LineWidth',gwLineWidth,'LineStyle','-');

tau_cha = tau_zheng_save - tau_0_save;
postion = find(tau_cha>1e-3);
postion = [postion(1)-1,postion];
tau_0_save = tau_0_save(postion);
tau_zheng_save = tau_zheng_save(postion);
d_ss = d_ss(postion);

plot(tau_0_save, d_ss,'r-','LineWidth',gwLineWidth)
plot(tau_zheng_save, d_ss,'b-','LineWidth',gwLineWidth)

x_fill = [tau_0_save, fliplr(tau_zheng_save)]; 
y_fill = [d_ss,fliplr(d_ss)]; 

f2 = fill(x_fill, y_fill, 'r', 'FaceAlpha', 0.2,'EdgeColor', 'b','LineWidth',gwLineWidth,'LineStyle','-'); 


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

text(27.2801047120419, 14.0461538461538,{'spinodal region'},'FontSize',18,'FontName','Times New Roman',...
    'HorizontalAlignment','center','Interpreter','latex');

text(33.4162303664922, 6.93846153846151,{'binodal region'},'FontSize',18,'FontName','Times New Roman',...
    'HorizontalAlignment','center','Rotation',25,'Interpreter','latex');

text(11.3010471204188, 16.7538461538461,{'binodal', 'region'},'FontSize',18,'FontName','Times New Roman',...
    'HorizontalAlignment','center','Rotation',45,'Interpreter','latex');

box on
N_max = 40;
 
xlim([0 N_max])
ylim([0 20])

set(gca,'XColor','k','YColor','k','TickLength',...
    [0.02 0.02],'FontSize',24,'linewidth',gwLineWidth,'layer','top');

figure_name = ['./FigureS7b_binodalregion.eps'];
saveas(gcf, figure_name, 'epsc');

