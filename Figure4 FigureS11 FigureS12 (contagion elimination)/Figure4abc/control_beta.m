clear all
close all
clc

beta1 = 0.01; nu = 0.2; gamma = 1 / 7; 

% control beta
beta = beta1/4; 

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
[sol_t, sol_x] = ode45(@(t,x) OdeSISystem(t,x,beta1,nu,gamma), tspan, x0);
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

set(gca,'XColor','m','YColor','m','TickLength',...
    [0.02 0.02],'FontSize',24,'linewidth',gwLineWidth,'layer','top');



%%


control_N1 = 13.36;
S = endingS;
I = control_N1 - S;

plot(control_N1, I, 'MarkerSize',18,'Marker','.','LineStyle','none','Color','k');

figure_name = ['./beijing control  beta ' 'beta=' num2str(beta1,'%.3f') ' nu=' num2str(nu,'%.3f')  '.eps'];
saveas(gcf, figure_name, 'epsc');


disp(['segreation I = ' num2str(endingI-I)])
disp(['S = ' num2str(S) ', I = ' num2str(I) ', S+I = ' num2str(S+I)])
x0 = [S; I];
[controlbeta_t, controlbeta_x] = ode45(@(t,x) OdeSISystem(t,x,beta,nu,gamma), tspan, x0);

figure
set(gcf,'Position', [100 300 500 420])
axes('Position',[0.169 0.225 0.758 0.725]);
set(gca,'Color','none')
hold on
plot(sol_t,      sol_x(:,2),'LineWidth',gwLineWidth,'LineStyle','-','Color','r');
plot(controlbeta_t+sol_t(end), controlbeta_x(:,2),'LineWidth',gwLineWidth,'LineStyle','-','Color','b');

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


set(gca,'XColor','k','YColor','k','TickLength',...
    [0.02 0.02],'FontSize',24,'linewidth',gwLineWidth,'layer','top');

save databeijing_backward_bifurcation_control_beta controlbeta_t controlbeta_x