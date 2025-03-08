clear all
close all
clc

beta = 0.01; nu = 0.2; gamma = 1 / 7; 

gwLineWidth = 1.5;

load databeijing_backward_bifurcation_control_population
load databeijing_backward_bifurcation_control_beta
load databeijing_backward_bifurcation_control_nu

figure
set(gcf,'Position', [100 300 756 280])
axes('Position',[0.0608465608465608 0.157142857142857 0.908730158730159 0.786278195488721]);
set(gca,'Color','none')
hold on
p1 = plot(sol_t,      sol_x(:,2),'LineWidth',gwLineWidth,'LineStyle','-','Color','r');

p3 = plot(controlbeta_t+sol_t(end), controlbeta_x(:,2),'LineWidth',gwLineWidth,'LineStyle','-','Color','magenta');
p4 = plot(controlnu_t+sol_t(end), controlnu_x(:,2),'LineWidth',gwLineWidth,'LineStyle','-','Color',[0.850980392156863 0.325490196078431 0.0980392156862745]);
p2 = plot(controlI_t+sol_t(end), controlI_x(:,2),'LineWidth',gwLineWidth,'LineStyle','-','Color','blue');


y = linspace(sol_x(end,2), controlI_x(1,2), 3);
x = 600 * ones(size(y));
plot(x, y,'LineWidth',gwLineWidth,'LineStyle',':','Color','k');

cont_t = [sol_t(1:end-1); sol_t+sol_t(end)];
% cont_I = [sol_x(:,2), control1_x(:,2)];

ax = gca;  
ax.XTick = [0:200:1200]; 
ax.XTickLabel = [0:200:1200]-600; 
ax.XAxis.FontName = 'Times New Roman';  
ax.TickLabelInterpreter = 'latex';


ax.YTick = [0:2:10]; 
ax.YTickLabel = [0:2:10]; 
ax.YAxis.FontName = 'Times New Roman'; 
ax.TickLabelInterpreter = 'latex';

ax.TickDir = 'in';

% axis equal
xlim([0 600*2])
ylim([-0.50 8.5])


legend1 = legend([p2,p3,p4], ...
    {['isolate $I$'], ['decrease $\beta$'], ['decrease $\nu$']});

set(legend1,...
    'Position',[0.680392469270164 0.525536119590994 0.255454091576396 0.326428564276014],...
    'Orientation','vertical',...
    'Interpreter','latex',...
    'FontSize',18,...
    'FontName','Times New Roman',...
    'EdgeColor','none',...
    'Color','none');

box on

annotation('textbox',...
    [0.0681216931216906 0.771428571428569 0.0892857142857164 0.156142857142855],...
    'String','$I(t)$',...
    'Interpreter','latex',...
    'FontSize',24,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation('textbox',...
    [0.80753968253968 0.208928571428571 0.145502645502646 0.138285714285714],...
    'String',{'Time $t$'},...
    'Interpreter','latex',...
    'FontSize',24,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');

set(gca,'XColor','k','YColor','k','TickLength',...
    [0.01 0.01],'FontSize',24,'linewidth',gwLineWidth,'layer','top');

figure_name = ['./beijing control effect ' 'beta=' num2str(beta,'%.3f') ' nu=' num2str(nu,'%.3f')  '.eps'];
saveas(gcf, figure_name, 'epsc');


