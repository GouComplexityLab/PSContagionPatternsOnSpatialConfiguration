function out = my_draw_arrow1(startpoint,endpoint,colorname,lw)

% close all
% clear all
% hold on; 
% grid on;
% startpoint=[0,0];
% endpoint=[1,1];
s=sqrt((startpoint(1)-endpoint(1))^2+(startpoint(2)-endpoint(2))^2); % 标准化以使得箭头一样大
if s>1
    lambda1=1/s;
else
    lambda1=0.5;
end

% lambda1=0.2;

point0=lambda1.*startpoint+(1-lambda1).*endpoint;
v1=point0-endpoint;
theta = 90*pi/180; 
theta1 = -1*90*pi/180; 
rotMatrix = [cos(theta)  -sin(theta) ; sin(theta)  cos(theta)];
rotMatrix1 = [cos(theta1)  -sin(theta1) ; sin(theta1)  cos(theta1)];  
v2 = v1*rotMatrix; 
v3 = v1*rotMatrix1; 

x1 = point0;
x2 = x1 + v2; 
x3 = x1 + v3; 
x1=endpoint;
lambda2=0.9;
x2=lambda2.*point0+(1-lambda2).*x2;
x3=lambda2.*point0+(1-lambda2).*x3;
% fill([x1(1) x2(1) x3(1)],[x1(2) x2(2) x3(2)],'r','LineStyle','none'); % this fills the arrowhead (black)

% % fill([x1(1) x2(1) x3(1)],[x1(2) x2(2) x3(2)],'k','EdgeColor',colorname); % this fills the arrowhead (black)
% % plot([startpoint(1) point0(1)],[startpoint(2) point0(2)],'linewidth',1,'color',colorname);

plot([startpoint(1) endpoint(1)],[startpoint(2) endpoint(2)],'linewidth',lw,'color',colorname);


% axis equal

