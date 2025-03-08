clear all
close all
clc

% gamma = 1/3; beta = 0.05; nu = 2; d_s = 2; d_i = 1; 
beta = 0.01; nu = 0.2/20; gamma = 1 / 7; d_s = 4; d_i = 1;

disp(['judge ' num2str(gamma/beta - 1/nu)])
disp(['judge ' num2str(gamma/beta - d_i/4/d_s)])
disp(['p_star ' num2str(-d_i + 2*sqrt(d_s*d_i*gamma/beta))])


disp(['judge condition di/ds - gamma*nu/beta :' num2str(d_i/d_s - gamma*nu/beta)])
disp(['p_star_low ' num2str(2*sqrt(d_s*d_i*gamma/beta/nu) - d_i/nu)])
disp(['p_star_up ' num2str(d_s*gamma/beta)])

i_right = 70;

%%
i_T = 0;
s_T = gamma/beta;

i_1 = sqrt(gamma/beta/nu)-1/nu;
i_Fq = linspace(0,i_1);
s_Fq = gamma./(beta.*(1+nu.*i_Fq));

i_2 = sqrt(d_s/d_i)*(sqrt(gamma/beta/nu))-1/nu;
i_Fh = linspace(i_1,i_2,20);
s_Fh = gamma./(beta.*(1+nu.*i_Fh));

i_D = linspace(i_2,i_right);
s_D = gamma./(beta.*(1+nu.*i_D));

%%
figure
set(gcf,"Position",[100 100 900 560])
axes('Position',[0.137533875338753 0.159821428571429 0.842466124661246 0.801785714285714]);
hold on 
ic = i_right;
sc = gamma./(beta.*(1+nu.*ic));

k = - d_i/d_s;

p2 = d_s * gamma/beta;
i_values = linspace(0,i_right,500+1);  
l2_s = k * i_values + p2/d_s;
plot(i_values,l2_s,'k--','LineWidth',2.0)

p3 = 2*sqrt(d_s*d_i*gamma/beta/nu) - d_i/nu;

pc = 70;
ic_values = linspace(0,i_right,500+1);  
lc_s = k * ic_values + pc/d_s;
plot(ic_values,lc_s,'k-','LineWidth',1.0)

%%

plot(i_Fq,s_Fq,'r-','LineWidth',2.0)
plot(i_Fh,s_Fh,'r-','LineWidth',2.0)
plot(i_D,s_D,'r-','LineWidth',2.0)

up_s = 20;
plot(zeros(size(linspace(0,s_T,100+1))),linspace(0,s_T,100+1),'r-','LineWidth',2.0)
plot(zeros(size(linspace(s_T,up_s,100+1))),linspace(s_T,up_s,100+1),'r-','LineWidth',2.0)

plot(i_T,s_T,'b*','LineWidth',1.1)
disp(['i_T=' num2str(i_T) ', s_T=' num2str(s_T)])

i_F = sqrt(gamma/beta/nu)-1/nu;
s_F = gamma/(beta*(1+nu*i_F));
plot(i_F,s_F,'k*','LineWidth',1.1)
i_D = sqrt(d_s/d_i)*(sqrt(gamma/beta/nu))-1/nu;
s_D = gamma/(beta*(1+nu*i_D));
plot(i_D,s_D,'c*','LineWidth',1.1)
disp(['i_D=' num2str(i_D) ', s_D=' num2str(s_D)])

ax = gca; 
ax.XTick = [0,10:10:70]; 
ax.XTickLabel = []; 
ax.XAxis.FontName = 'Times New Roman'; 
ax.TickLabelInterpreter = 'latex';

ax.YTick = [0,4:4:20]; 
ax.YTickLabel = {'0', '', '', '', '', '', '', ''}; 
ax.YAxis.FontName = 'Times New Roman'; 
ax.TickLabelInterpreter = 'latex';

ax.XAxis.TickDirection = 'in';
ax.YAxis.TickDirection = 'in';


xlim([0 i_right])
ylim([0 up_s])
xlabel('$I$','FontSize',30,'Interpreter','latex')
ylabel('$S$','FontSize',30,'Interpreter','latex')

annotation(gcf,'textbox',...
    [0.138333333333333 0.733928571428571 0.0472222222222222 0.091048770984742],...
    'String',{'$T$'},'Interpreter','latex','FontSize',36,'Color', 'b',...
    'FitBoxToText', 'off', ...
    'EdgeColor', 'none');

annotation(gcf,'textbox',...
    [0.699249007936504 0.3875 0.269084325396829 0.120686164896414],...
    'String',{'$f(S,I)=0$'},...
    'FontSize',36,'Interpreter','latex','EdgeColor', 'none','Color', 'r',...
    'FitBoxToText','off');
annotation(gcf,'textbox',...
    [0.137089285714283 0.843749999999999 0.376799603174607 0.117658730158729],...
    'String',{'$d_{S}S+d_{I}I=P^{*}$'},...
    'Rotation',-27.5,...
    'FontSize',36,'Interpreter','latex','EdgeColor', 'none',...
    'FitBoxToText','off');

box('on');

gwLineWidth = 1.0;
set(gca,'XColor','k','YColor','k','TickLength',...
    [0.005 0.005],'FontSize',36,'linewidth',gwLineWidth,'layer','top');

figure_name = ['./FigureS3b_intersect.eps'];
saveas(gcf, figure_name, 'epsc');