clear all
close all
clc

beta = 0.01; nu = 0.2; gamma = 1 / 7;


% Max_N = 20;
Max_N = 40;
disp(['judge ' num2str(gamma/beta - 1/nu)])
N_1 = 2*sqrt(gamma/beta/nu)-1/nu
N_2 = gamma/beta

N_sn = linspace(0,N_2,1000);
I_0 = zeros(size(N_sn));

N_zheng = linspace(N_1,Max_N,500);
N_fu = linspace(N_1,N_2,500);
I_1 = (beta*(nu.*N_zheng - 1) + sqrt((beta+beta*nu.*N_zheng).^2-4*beta*nu*gamma))./(2*beta*nu);
I_2 = (beta*(nu.*N_fu - 1) - sqrt((beta+beta*nu.*N_fu).^2-4*beta*nu*gamma))./(2*beta*nu);

gwLineWidth = 1.5;
figure
set(gcf,'Position', [100 300 500 420])
axes('Position',[0.063 0.214285714285714 0.758 0.725]);
set(gca,'Color','none')

hold on

plot(linspace(0,Max_N,500),linspace(0,Max_N,500),'Color',[0.5 0.5 0.5],'LineWidth',gwLineWidth)

plot(real(N_fu),real(I_2),'k--','LineWidth',gwLineWidth);
plot(N_sn,I_0,'g-','LineWidth',gwLineWidth);

plot(linspace(N_2,Max_N,100+1),zeros(size(linspace(N_2,Max_N,100+1))),'k--','LineWidth',gwLineWidth);

plot(N_zheng,real(I_1),'r-','LineWidth',gwLineWidth);

N = 13.36;
I = 1.0;
S = N - I;
x0 = [S; I];
dt = 0.1; tspan = [0:dt:600];
[sol_t, sol_x] = ode45(@(t,x) OdeSISystem(t,x,beta,nu,gamma), tspan, x0);
endingS = sol_x(end,1);
endingI = sol_x(end,2);

s1 = 220;
scatter(N, endingI,s1,'MarkerFaceColor','r', ...
    'MarkerEdgeColor','r','Marker','pentagram');



set(gca, 'YAxisLocation', 'right');
ax = gca; 
ax.XTick = [0:10:Max_N]; 
ax.XTickLabel = {'0','10','20','30','$\langle N\rangle$'}; 
ax.XAxis.FontName = 'Times New Roman'; 
ax.TickLabelInterpreter = 'latex';


ax.YTick = [0:10:Max_N]; 
ax.YTickLabel = {'0','10','20','30','$I^{*}$'}; 
ax.YAxis.FontName = 'Times New Roman'; 
ax.TickLabelInterpreter = 'latex';

ax.TickDir = 'out';

xlim([0 Max_N])
ylim([-0.5,Max_N])

set(gca,'XColor','b','YColor','b','TickLength',...
    [0.02 0.02],'FontSize',24,'linewidth',gwLineWidth,'layer','top');

%%


control_N1 = 7.664;
S = endingS;
I = control_N1 - S;

plot(control_N1, I, 'MarkerSize',18,'Marker','.','LineStyle','none','Color','k');



disp(['segreation I = ' num2str(endingI-I)])
disp(['S = ' num2str(S) ', I = ' num2str(I) ', S+I = ' num2str(S+I)])
x0 = [S; I];
[controlI_t, controlI_x] = ode45(@(t,x) OdeSISystem(t,x,beta,nu,gamma), tspan, x0);

point1 = [13.36, endingI];
point2 = [control_N1, I];

gwHeadWidth = 8;
gwcolor_h = 'k';
gwcolor_v = [0.16078431372549 0.501960784313725 0.364705882352941];
gwLineStyle_h = ':';
gwLineStyle_v = '-';

axPos = ax.Position; 
xLimits = get(gca, 'XLim');
yLimits = get(gca, 'YLim');

startX = point1(1); startY = point1(2); 
endX = point2(1);   endY = point2(2); 

relativeStartX = axPos(1) + (startX - xLimits(1)) / (xLimits(2) - xLimits(1)) * axPos(3);
relativeStartY = axPos(2) + (startY - yLimits(1)) / (yLimits(2) - yLimits(1)) * axPos(4);
relativeEndX = axPos(1) + (endX - xLimits(1)) / (xLimits(2) - xLimits(1)) * axPos(3);
relativeEndY = axPos(2) + (endY - yLimits(1)) / (yLimits(2) - yLimits(1)) * axPos(4);
annotation('arrow', [relativeStartX, relativeEndX], [relativeStartY, relativeEndY], ...
    'LineWidth',gwLineWidth,'HeadWidth',gwHeadWidth,'HeadLength',gwHeadWidth, ...
    'Color',gwcolor_h,'LineStyle',gwLineStyle_h);

figure_name = ['./beijing control  I ' 'beta=' num2str(beta,'%.3f') ' nu=' num2str(nu,'%.3f')  '.eps'];
saveas(gcf, figure_name, 'epsc');

figure
set(gcf,'Position', [100 300 500 420])
axes('Position',[0.187 0.225 0.758 0.725]);
set(gca,'Color','none')
hold on
plot(sol_t,      sol_x(:,2),'LineWidth',gwLineWidth,'LineStyle','-','Color','r');
plot(controlI_t+sol_t(end), controlI_x(:,2),'LineWidth',gwLineWidth,'LineStyle','-','Color','b');

ax = gca; 
ax.XTick = [0:200:tspan(end)]; 
ax.XTickLabel = [0:200:tspan(end)]; 
ax.XAxis.FontName = 'Times New Roman'; 
ax.TickLabelInterpreter = 'latex';


ax.YTick = [0:2:10]; 
ax.YTickLabel = [0:2:10]; 
ax.YAxis.FontName = 'Times New Roman';
ax.TickLabelInterpreter = 'latex';

ax.TickDir = 'out';


xlim([0 tspan(end)*2])
ylim([0 10])

xlabel('Time $t$','FontName','Times New Roman','FontSize',24,'Interpreter','latex');
ylabel('$I(t)$','FontName','Times New Roman','FontSize',24,'Interpreter','latex');

set(gca,'XColor','k','YColor','k','TickLength',...
    [0.02 0.02],'FontSize',24,'linewidth',gwLineWidth,'layer','top');

save databeijing_backward_bifurcation_control_population controlI_t controlI_x sol_t sol_x

