clear all
close all
clc

beta = 0.01; nu = 0.2; gamma = 1 / 7; 

Max_N = 26;
disp(['judge ' num2str(gamma/beta - 1/nu)])
N_1 = 2*sqrt(gamma/beta/nu)-1/nu;
N_2 = gamma/beta;

N_sn = linspace(0,N_2,1000);
I_0 = zeros(size(N_sn));

N_zheng = linspace(N_1,Max_N,500);
N_fu = linspace(N_1,N_2,500);
I_1 = (beta*(nu.*N_zheng - 1) + sqrt((beta+beta*nu.*N_zheng).^2-4*beta*nu*gamma))./(2*beta*nu);
I_2 = (beta*(nu.*N_fu - 1) - sqrt((beta+beta*nu.*N_fu).^2-4*beta*nu*gamma))./(2*beta*nu);

gwLineWidth = 1.5;
figure
set(gcf,'Position', [100 300 500 420])
axes('Position',[0.06 0.15 0.747 0.775000000000001]);
set(gca,'Color','none')

hold on

x1 = linspace(0,N_1,500); y1 = linspace(0,N_1,500);
x2 = N_fu; y2 = I_2;
x = [x1,x2]; 
y_up = [y1,y2];
y_low = zeros(size(y_up));

fill_x = [x, fliplr(x)];
fill_y = [y_up, fliplr(y_low)];

x1 = real(N_fu); y1 = real(I_2);
x2 = linspace(N_2,Max_N,100+1); y2 = zeros(size(linspace(N_2,Max_N,100+1)));
x = [x1,x2]; 
y_low = [y1,y2];
y_up = linspace(N_1,Max_N,500);

fill_x = [linspace(N_1,Max_N,500), fliplr(x)];
fill_y = [y_up, fliplr(y_low)];

plot(linspace(0,Max_N,500),linspace(0,Max_N,500),'Color',[0.5 0.5 0.5],'LineWidth',gwLineWidth)

plot(real(N_fu),real(I_2),'k--','LineWidth',gwLineWidth);
plot(N_sn,I_0,'k-','LineWidth',gwLineWidth);

plot(linspace(N_2,Max_N,100+1),zeros(size(linspace(N_2,Max_N,100+1))),'k--','LineWidth',gwLineWidth);

plot(N_zheng,real(I_1),'k-','LineWidth',gwLineWidth);





y = linspace(0,N_1,3);
x = N_1* ones(size(y));
plot(x,y,'k:','LineWidth',gwLineWidth,'Color',[0.5 0.5 0.5])

x = linspace(N_1,Max_N,3);
y = N_1* ones(size(x));
plot(x,y,'k:','LineWidth',gwLineWidth,'Color',[0.5 0.5 0.5])

y = linspace(0,N_2,3);
x = N_2* ones(size(y));
plot(x,y,'k:','LineWidth',gwLineWidth,'Color',[0.5 0.5 0.5])

x = linspace(N_2,Max_N,3);
y = N_2* ones(size(x));
plot(x,y,'k:','LineWidth',gwLineWidth,'Color',[0.5 0.5 0.5])

plot(N_2,0,'r.','MarkerSize',18,'LineWidth',2.0);
plot(N_1,real(I_1(1)),'b.','MarkerSize',18,'LineWidth',2.0);


set(gca, 'YAxisLocation', 'right');
ax = gca; 
ax.XTick = [0,  N_2]; 
ax.XTickLabel = {'0', '$N_{c}$'}; 
ax.XAxis.FontName = 'Times New Roman';
ax.TickLabelInterpreter = 'latex';


ax.YTick = [0,N_1, N_2]; 
ax.YTickLabel = {'0','$N^{sub}_{c}$', '$N_{c}$'};  
ax.YAxis.FontName = 'Times New Roman';
ax.TickLabelInterpreter = 'latex';

ax.TickDir = 'out';

xlim([0 Max_N+0.])
ylim([-0.3,Max_N])
set(gca,'linewidth',3,'FontSize',30)

annotation('arrow',[0.771-0.06 0.869-0.06],...
    [0.12 0.12],'LineWidth',2,'HeadWidth',12,...
    'HeadLength',12);
annotation('textbox',...
    [0.761-0.06 0.0035714285714286 0.089 0.113095238095238],...
    'Color','k',...
    'String',{'$N$'},...
    'FontName','Times New Roman',...
    'Interpreter','latex',...
    'FontSize',28,...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');

annotation('arrow',[0.896-0.06 0.896-0.06],...
    [0.826190476190476 0.927380952380952],'LineWidth',2,'HeadWidth',12,...
    'HeadLength',12);
annotation('textbox',...
    [0.902000000000004-0.06 0.817857142857149 0.0639999999999972 0.12519047619048],...
    'Color','k',...
    'String',{'$I^{*}$'},...
    'FontName','Times New Roman',...
    'Interpreter','latex',...
    'FontSize',28,...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'BackgroundColor','none');

annotation(gcf,'textbox',...
    [0.460614087301587 0.164285714285716 0.0933859126984141 0.0651914908617374],...
    'String',{'TB'},'Interpreter','latex','FontSize',18,'Color', 'r',...
    'FitBoxToText', 'off', ...
    'EdgeColor', 'none');

annotation(gcf,'textbox',...
    [0.282614087301587 0.227380952380953 0.112385912698413 0.0747153003855492],...
    'String',{'SNB'},'Interpreter','latex','FontSize',18,'Color', 'b',...
    'FitBoxToText', 'off', ...
    'EdgeColor', 'none');

set(gca,'XColor','k','YColor','k','TickLength',...
    [0.02 0.02],'FontSize',24,'linewidth',gwLineWidth,'layer','top');

figure_name = ['./FigureS2a_backward.eps'];
saveas(gcf, figure_name, 'epsc');
