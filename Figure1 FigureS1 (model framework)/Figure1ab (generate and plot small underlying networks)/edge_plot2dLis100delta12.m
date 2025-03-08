
clc
close all
clear all

L = 100;

data_path1 = ['./Generated delta Lattice Embedded Networks L=' num2str(L) ' dg=12'];
data_path2 = ['/mat_matrices'];
mat_name = ['/matrices_deltaLEN_L=' num2str(L) '_dg=12_number=1.mat'];
net_name = ['delta12L' num2str(L)];
show_step = 4;

%%
data_pathname = strcat(data_path1,data_path2,mat_name);

load([data_pathname])
weight_adjacent = full(adjacent_matrix); 

Nx = sqrt(size(weight_adjacent, 1));
Ny = sqrt(size(weight_adjacent, 2));

[x_2d, y_2d] = meshgrid((1:Nx)-0.5, (1:Ny)-0.5);

D = Nx;
%%
x = x_2d(:); y = y_2d(:);

hfig=figure('Color',[1 1 1]);
set(0,'DefaultFigureVisible', 'on')
set(gcf,'Position', [100 100 600 600]);
hold on
lw = 1.5;
plot([0,0],[0,D],'linewidth',lw,'color',[0.5 0.5 0.5 ])
plot([0,D],[D,D],'linewidth',lw,'color',[0.5 0.5 0.5 ])
plot([D,D],[0,D],'linewidth',lw,'color',[0.5 0.5 0.5 ])
plot([D,0],[0,0],'linewidth',lw,'color',[0.5 0.5 0.5 ])


colorname = 'k'; lw = 0.3;
Number_nodes = size(weight_adjacent,1);
for i=1:Number_nodes
    disp(['i = ' num2str(i)])
      for j=find(weight_adjacent(i,:)==1)
          startpoint = [x(j) y(j)];
          endpoint = [x(i) y(i)];
          [distance position_cases] = distance_computing(x(i),y(i),x(j),y(j),D);
          R = distance * 1.5;
          plot_PBc(startpoint,endpoint,R,D,colorname, lw);
      end
end        

central_nodex = fix(Nx/2); central_nodey = fix(Ny/2);
% central_nodex = 4; central_nodey = 1;
central_point_index = conv_coordinate2d_to_index(central_nodex, central_nodey, Ny );

colorname = 'b'; lw = 1.5;
for i=central_point_index
    disp(['i = ' num2str(i)])
      for j=find(weight_adjacent(i,:)==1)
          startpoint = [x(j) y(j)];
          endpoint = [x(i) y(i)];
          [distance position_cases] = distance_computing(x(i),y(i),x(j),y(j),D);
          R = distance * 1.5;
          plot_PBc(startpoint,endpoint,R,D,colorname, lw);
      end
end   


scatter(x,y,10,'MarkerEdgeColor','k',...
    'MarkerFaceColor','k',...
    'Marker','.')

center_point_color = 'r';
plot(central_nodex-0.5,central_nodey-0.5,'LineStyle','none',...
    'MarkerSize',3,'Marker','pentagram','LineWidth',1.5,...
    'MarkerFaceColor',center_point_color,'MarkerEdgeColor',center_point_color);

axis equal
axis off
% view(3)
% view([-27 85])
%%


x_low = central_nodex-show_step;
x_top = central_nodex+show_step-1;
y_low = central_nodey-show_step;
y_top = central_nodey+show_step-1;
lw = 1.0; colorname= [1 0.411764705882353 0.16078431372549];
plot([x_low,x_low],[y_low,y_top],'linewidth',lw,'color',colorname);
plot([x_top,x_low],[y_top,y_top],'linewidth',lw,'color',colorname);
plot([x_top,x_top],[y_top,y_low],'linewidth',lw,'color',colorname);
plot([x_low,x_top],[y_low,y_low],'linewidth',lw,'color',colorname);

set(gca, 'LooseInset', [0.01,0.01,0.01,0.01]);

eps_name = [net_name '.eps'];
print(eps_name,'-depsc')
saveas(gcf,[net_name '.jpg'])
disp('save eps done!')
%%

weight_central_point = weight_adjacent(:, central_point_index);
weight_central_point2 = weight_adjacent(central_point_index, :)';
sum(weight_central_point)
disp(['check symmestry: ' num2str(sum(abs(weight_central_point-weight_central_point2)))])

matrix_weight_central_point = reshape(weight_central_point, Nx, Ny);
sum(matrix_weight_central_point(:))


%%
hfig=figure('Color',[1 1 1]);
set(0,'DefaultFigureVisible', 'on')
set(gcf,'Position', [100    100    600    600]);
hold on

%%
zero_indices = find(matrix_weight_central_point==0);

x_zero = x_2d(zero_indices);
y_zero = y_2d(zero_indices);
z_zero = 0.002*ones(size(x_zero));
points_color = 'c';
points_color = [0.5 0.5 0.5];
plot3(x_zero, y_zero,z_zero,'LineStyle','none',...
    'MarkerSize',12,'Marker','.','LineWidth',1.5,...
    'MarkerFaceColor',points_color,'MarkerEdgeColor',points_color);

%%
one_indices = find(matrix_weight_central_point==1);
x_one = x_2d(one_indices);
y_one = y_2d(one_indices);
z_one = 0.002*ones(size(x_one));
points_color = [0 0 1];
plot3(x_one, y_one,z_one,'LineStyle','none',...
    'MarkerSize',18,'Marker','.','LineWidth',1.5,...
    'MarkerFaceColor',points_color,'MarkerEdgeColor',points_color);

%%
center_point_color = 'r';
plot3(central_nodex-0.5,central_nodey-0.5,0.002*1,'LineStyle','none',...
    'MarkerSize',6,'Marker','pentagram','LineWidth',1.5,...
    'MarkerFaceColor',center_point_color,'MarkerEdgeColor',center_point_color);

%%
lw = 2; lc = 'k';

h = 1;
xc = central_nodex-0.5; yc = central_nodey-0.5;

for each_one = 1:length(one_indices)
    xn = x_one(each_one); yn = y_one(each_one);
    point1 = [xc,yc,0]; point2 = [xn,yn,0];
    h = min(abs(xn-xc),abs(L+xn-xc)) + min(abs(yn-yc),abs(L+yn-yc))*0.5;
    plot_parabola_through_points(point1,point2,h,lw,lc)
end

%%

x_low = central_nodex-show_step;
x_top = central_nodex+show_step-1;
y_low = central_nodey-show_step;
y_top = central_nodey+show_step-1;
lw = 1.0; colorname= [1 0.411764705882353 0.16078431372549];
plot([x_low,x_low],[y_low,y_top],'linewidth',lw,'color',colorname);
plot([x_top,x_low],[y_top,y_top],'linewidth',lw,'color',colorname);
plot([x_top,x_top],[y_top,y_low],'linewidth',lw,'color',colorname);
plot([x_low,x_top],[y_low,y_low],'linewidth',lw,'color',colorname);

%%
axis equal
axis off
% 
% show_step = 3;
xlim([central_nodex-show_step central_nodex+show_step-1])
ylim([central_nodey-show_step central_nodey+show_step-1])

view([-27 30])

set(gca, 'LooseInset', [0.01,0.01,0.01,0.01]);

eps_name = [net_name 'part.eps'];
print(eps_name,'-depsc')
saveas(gcf,[net_name 'part.jpg'])
disp('save eps done!')
