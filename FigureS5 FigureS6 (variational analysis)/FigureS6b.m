clear all
close all
clc

gamma = 1/3; beta = 0.05; nu = 2; d_s = 16; d_i = 1; 

beta = 0.01; nu = 0.2; gamma = 1 / 7; d_s = 4; d_i = 1; 


disp(['judge condition di/ds - gamma*nu/beta :' num2str(d_i/d_s - gamma*nu/beta)])
disp(['p_star_low ' num2str(2*sqrt(d_s*d_i*gamma/beta/nu) - d_i/nu)])
disp(['p_star_up ' num2str(d_s*gamma/beta)])

p_star_low = 2*sqrt(d_s*d_i*gamma/beta/nu) - d_i/nu;
p_star_up = d_s*gamma/beta;
%% find Maxwell point

min_cha = [];
p_stars = linspace(p_star_low,p_star_up,100000+1);
p_stars = linspace(29.8,29.9,10e4+1);
p_stars = [29.802416:1e-10:29.802418];

%%
p_star_maxwell = 29.8024171966;  
p_star = p_star_maxwell;
% p_star = 30; % p_low<p_star<p_star_maxwell<p_up
% p_star = 12.0; % <p_low
% p_star = 122.0; % >p_up

%% 
q_0 = p_star;
disp(['q0:    ' num2str(q_0,'%.10f') ])
s0 = 1/2/d_s*(q_0 + p_star);
i0 = 1/2/d_i*(p_star - q_0);
disp(['(s0, i0):    (' num2str(s0,'%.10f')  ', ' num2str(i0,'%.10f') ')' ])

delta_q = (2*beta*d_i)^2 + 4*beta*nu*(2*d_i*beta*p_star + beta*nu*(p_star)^2 - 4*d_s*d_i*gamma);
q_u = d_i/nu + (-1/2/beta/nu)*(delta_q)^(1/2);
disp(['q_u:    ' num2str(q_u,'%.10f') ])
s_u = 1/2/d_s*(q_u + p_star);
i_u = 1/2/d_i*(p_star - q_u);
disp(['(s_u, i_u):    (' num2str(s_u,'%.10f')  ', ' num2str(i_u,'%.10f') ')' ])

q_d = d_i/nu - (-1/2/beta/nu)*(delta_q)^(1/2);
disp(['q_d:    ' num2str(q_d,'%.10f') ])
s_d = 1/2/d_s*(q_d + p_star);
i_d = 1/2/d_i*(p_star - q_d);
disp(['(s_d, i_d):    (' num2str(s_d,'%.10f')  ', ' num2str(i_d,'%.10f') ')' ])

q_pre = [q_0,q_d,q_u];
disp(['===>p_star=' num2str(p_star,'%.10f') ', delta_q=' num2str(delta_q) ', [q_0,q_d,q_u]=: ' num2str(q_pre,'%.4f')])
%%
Hamilton = @(p,q) (d_s + d_i).*(q.^2.*(gamma./(4.*d_i) - (beta.*nu.*p_star.^2)/(16.*d_i.^2.*d_s)) - q.*((gamma.*p_star)./(2*d_i) - (beta.*p_star.^2.*((nu.*p_star)/(2.*d_i) + 1))./(4.*d_i.*d_s)) - (beta.*q.^3.*((nu.*p_star)/(2.*d_i) + 1))/(12.*d_i.*d_s) + (beta.*nu.*q.^4)./(32.*d_i.^2.*d_s));


H0 = Hamilton(p_star,q_0);
Hd = Hamilton(p_star,q_d);
Hu = Hamilton(p_star,q_u);
disp(['H0=: ' num2str(H0,'%.10f')  ',  Hd=: ' num2str(Hd,'%.10f') ',  Hu=: ' num2str(Hu,'%.10f')])

disp(['Hu-Hd=: ' num2str(Hu-Hd,'%.10f') ])

disp(['H0=: ' num2str(H0,'%.10f')  ',  Hu=: ' num2str(Hu,'%.10f') ',  Hd=: ' num2str(Hd,'%.10f')])

disp(['p_star=' num2str(p_star,'%.10f')  ', H0-Hd=: ' num2str(H0-Hd,'%.10f')  ',  Hu-Hd=: ' num2str(Hu-Hd,'%.10f') ',  H0-Hu=: ' num2str(H0-Hu,'%.10f') ])

disp(['(H0-Hd)-(Hu-Hd)=: ' num2str((H0-Hd)-(Hu-Hd),'%.10f') ])

disp(['p_star=' num2str(p_star,'%.10f') ',  H0=: ' num2str(H0,'%.10f')  ',  Hd=: ' num2str(Hd,'%.10f')  ',  Hu=: ' num2str(Hu,'%.10f')]);

q = linspace(-8,35,500);
disp(['q_0 = ' num2str(q_0) 'q_u = ' num2str(q_u) 'q_d = ' num2str(q_d)])

Hq = Hamilton(p_star,q);
%%
fun_g = @(p_star,q) beta.*(1 + nu.*(p_star - q)./2./d_i).*((p_star).^2 - q.^2)./4./d_s./d_i - gamma.*(p_star - q)./2./d_i;
g = fun_g(p_star,q);

g0 = fun_g(p_star,q_0);
gd = fun_g(p_star,q_d);
gu = fun_g(p_star,q_u);



%%
figure
set(gcf,"Position",[100 100 900 560])
axes('Position',[0.176111111111111 0.13 0.630555555555556 0.8]);
hold on
yyaxis left
plot(q,zeros(size(q)),'k-.')

%% P*<P**
p1 = plot(q,g,'-','LineWidth',1.5);
plot(q_0,g0,'r.','MarkerSize',24,'LineWidth',2.0)
plot(q_d,gd,'b.','MarkerSize',24,'LineWidth',2.0)
plot(q_u,gu,'.','MarkerSize',24,'LineWidth',2.0,'Color', [0.494117647058824 0.184313725490196 0.556862745098039])

set(gca, 'XAxisLocation', 'origin');
xlim([min(q),max(q)])


x1 = linspace(q_u, q_d, 100);
x1_fill = [x1, fliplr(x1)]; 
y1 = fun_g(p_star,x1);
y1_fill = [y1,zeros(size(y1))]; 

fill(x1_fill, y1_fill, 'c', 'FaceAlpha', 0.3,'EdgeColor', 'none'); 

hold on
x2 = linspace(q_d, q_0, 100);
x2_fill = [x2, fliplr(x2)]; 
y2 = fun_g(p_star,x2);
y2_fill = [y2,zeros(size(y2))];  

fill(x2_fill, y2_fill, 'r', 'FaceAlpha', 0.3,'EdgeColor', 'none'); 

set(gca,'linewidth',3,'FontSize',30)
set(gca,'FontSize',30,'LineWidth',3);

xlabel('$Q$','FontSize',30,'Interpreter','latex')
ylabel('$g(P^{*},Q)$','FontSize',30,'Interpreter','latex')

ylim([-0.15 0.25])
ax = gca;
ax.YTick = [-0.1,0,0.1,0.2,0.4]; 
ax.YTickLabel = [-0.1,0,0.1,0.2,0.4]; 
ax.YAxis(1).FontName = 'Times New Roman';


hold on
yyaxis right

p2 = plot(q,Hq,'LineWidth',1.5,'Color',[0.850980392156863 0.325490196078431 0.0980392156862745]);

plot(q_0,H0,'r.','MarkerSize',24,'LineWidth',2.0)
plot(q_d,Hd,'b.','MarkerSize',24,'LineWidth',2.0)
plot(q_u,Hu,'.','MarkerSize',24,'LineWidth',2.0,'Color', [0.494117647058824 0.184313725490196 0.556862745098039])
xlim([min(q),max(q)])

xlabel('$Q$','FontSize',30,'Interpreter','latex')
ylabel('$V(P^{*},Q)$','FontSize',30,'Interpreter','latex')


ylim([-1.5,6.0])

ax = gca; 
ax.XTick = [0:10:30]; 
ax.XTickLabel = [0:10:30]; 
ax.XAxis.FontName = 'Times New Roman';
ax.TickLabelInterpreter = 'latex';
ax.XAxis.TickDirection = 'both';

ax.YTick = [-2:2:6]; 
ax.YTickLabel = [-2:2:6];


ax.TickLabelInterpreter = 'latex';

ax.YAxis(1).Color = 'blue'; 
ax.YAxis(2).Color = [1 0.411764705882353 0.16078431372549]; 

ax.YAxis(1).FontName = 'Times New Roman'; 
ax.YAxis(2).FontName = 'Times New Roman'; 
ax.YAxis(1).TickDirection = 'in';
ax.YAxis(2).TickDirection = 'in';

annotation('textbox',...
    [0.691275346817058 0.434563492063493 0.0731690976273861 0.11],...
    'String','$Q_{0}$',...
    'Interpreter','latex',...
    'FontSize',36,...
    'FitBoxToText', 'off', ...
    'EdgeColor', 'none','Color', 'r') 

annotation('textbox',...
    [0.472222222222222 0.430992063492064 0.0811111111111111 0.11],...
    'String','$Q_{-}$',...
    'Interpreter','latex',...
    'FontSize',36,...
    'FitBoxToText', 'off', ...
    'EdgeColor', 'none','Color', 'b') 

annotation('textbox',...
    [0.202777777777778 0.428710317460318 0.0827777777777775 0.11],...
    'String','$Q_{+}$',...
    'Interpreter','latex',...
    'FontSize',36,...
    'FitBoxToText', 'off', ...
    'EdgeColor', 'none','Color', [0.494117647058824 0.184313725490196 0.556862745098039]) 

gwLineWidth = 1.5;
set(gca,'TickLength', [0.02 0.02],'FontSize',36,'linewidth',gwLineWidth,'layer','top');

figure_name = ['./FigureS6b.eps'];
saveas(gcf, figure_name, 'epsc');