% �������ڱ߽�������������ľ���

function [distance position_cases]=distance_computing(x1,y1,x2,y2,D)

% D=30;r=1;
% % x1=1;y1=2; x2 = rand(1,1)*D;y2 = rand(1,1)*D;
% Is_in_circle(x1,y1,x2,y2,D,r);
x=[x1,x2];y=[y1,y2];
i=1;j=2;
DisTemp=zeros(9,1);
DisTemp(1)=sqrt((x(i)-x(j))^2+(y(i)-y(j))^2);% �� 1 ��
DisTemp(2)=sqrt((x(i)+D-x(j))^2+(y(i)-y(j))^2);% �� 2 ��
DisTemp(3)=sqrt((x(i)-x(j))^2+(y(i)+D-y(j))^2); % �� 3 ��
DisTemp(4)=sqrt((x(i)+D-x(j))^2+(y(i)+D-y(j))^2); % �� 4 ��
DisTemp(5)=sqrt((x(i)-D-x(j))^2+(y(i)-y(j))^2); % �� 5 ��
DisTemp(6)=sqrt((x(i)-x(j))^2+(y(i)-D-y(j))^2); % �� 6 ��
DisTemp(7)=sqrt((x(i)-D-x(j))^2+(y(i)-D-y(j))^2); % �� 7 ��
DisTemp(8)=sqrt((x(i)+D-x(j))^2+(y(i)-D-y(j))^2); % �� 8 ��
DisTemp(9)=sqrt((x(i)-D-x(j))^2+(y(i)+D-y(j))^2); % �� 9 ��
[Dis index]=min(DisTemp);
DisTemp;
distance=Dis;
position_cases = index;
% [sqrt((x(i)-x(j))^2+(y(i)-y(j))^2) Dis index]
clear i j DisTemp

% hold on
% % ����߽�
% line([0 0],[0 D],'Color','k','LineWidth',1.5);%����
% line([0 D],[0 0],'Color','k','LineWidth',1.5);%����
% line([0 D],[D D],'Color','k','LineWidth',1.5);%����
% line([D D],[0 D],'Color','k','LineWidth',1.5);%���� 
% % ���ڱ߽�
% 
% line([0+r 0+r],[0+r D-r],'Color','g','LineWidth',1.5);%����
% line([0+r D-r],[0+r 0+r],'Color','g','LineWidth',1.5);%����
% line([0+r D-r],[D-r D-r],'Color','g','LineWidth',1.5);%����
% line([D-r D-r],[0+r D-r],'Color','g','LineWidth',1.5);%���� 
% 
% % ����߽�
% 
% line([0-r 0-r],[0-r D+r],'Color','b','LineWidth',1.5);%����
% line([0-r D+r],[0-r 0-r],'Color','b','LineWidth',1.5);%����
% line([0-r D+r],[D+r D+r],'Color','b','LineWidth',1.5);%����
% line([D+r D+r],[0-r D+r],'Color','b','LineWidth',1.5);%���� 
% 
% 
% plot(x(1),y(1),'r*')
% plot(x(2),y(2),'gd')
% 
% % scatter(X,Y)
% axis([-0.05*D-r 1.05*D+r -0.05*D-r 1.05*D+r])
% % title('Susceptibility Radius','FontWeight','bold','FontSize',14,'FontName','Times New Roman')
% set(gca,'xtick',[0:5:D]);
% set(gca,'ytick',[0:5:D]);
% % grid on