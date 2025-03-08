function plot_parabola_through_points(point1,point2,h,lw,lc)

% 定义三个点的坐标
x0 = point1(1); y0 = point1(2); z0 = point1(3);
x1 = point2(1); y1 = point2(2); z1 = point2(3);
x2 = (x0+x1)/2; y2 = (y0+y1)/2; z2 = h;

xx1 = x1 - x0; yy1 = y1 - y0; zz1 = z1 - z0; p1 = sqrt(xx1^2 + yy1^2);
xx2 = x2 - x0; yy2 = y2 - y0; zz2 = z2 - z0; p2 = sqrt(xx2^2 + yy2^2);

% 计算系数a和b

A = [0, 0, 1; p1^2, p1, 1; p2^2, p2, 1];
B = [0; zz1; zz2];
sol_gw = linsolve(A,B);
a = sol_gw(1);
b = sol_gw(2);
c = sol_gw(3);

% 生成参数t，通常范围为0到1
t = linspace(0, 1, 10000); % 这里使用100个点
% 计算直线参数
if x0 ~= x1
    m_line = (y1 - y0) / (x1 - x0); % 计算斜率
    b_line = y0 - m_line * x0; % 计算截距
    
    % 使用参数方程计算直线上的点
    x_line_xy = x0 + t * (x1 - x0);
    y_line_xy = m_line * x_line_xy + b_line;
else
    % 使用参数方程计算直线上的点
    x_line_xy = x0 * ones(size(t));
    y_line_xy = y0 + t * (y1 - y0);
end

z_line_xy = 0*ones(size(x_line_xy));

xx_line_xy = x_line_xy - x0;
yy_line_xy = y_line_xy - y0;
p1_line_xy = sqrt(xx_line_xy.^2 + yy_line_xy.^2);
zz_line_xy = a*p1_line_xy.^2+b*p1_line_xy+c;
zz_line_xy = zz_line_xy + z0;

% 绘制抛物线
% figure;
hold on
% plot3(x_line_xy,y_line_xy, z_line_xy, lc, 'LineWidth', lw);
% plot3(x_line_xy,y_line_xy,zz_line_xy, lc, 'LineWidth', lw);

% plot3(x0, y0, z0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b'); % 标记起点
% plot3(x1, y1, z1, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b'); % 标记起点
% plot3(x2, y2, z2, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b'); % 标记起点

plot3(x_line_xy,y_line_xy,z_line_xy, 'b', 'LineWidth', lw);
% plot3(x_line_xy,y_line_xy,zz_line_xy, 'b', 'LineWidth', lw);
scatter3(x_line_xy,y_line_xy,zz_line_xy,10*(t+t(2)), t+t(2), 'filled',...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor','k');

xlabel('X');
ylabel('Y');
zlabel('Z');
grid on;

view(3)




end