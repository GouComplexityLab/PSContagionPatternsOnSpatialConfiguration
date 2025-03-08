function plot_PBc(startpoint,endpoint,r0,D,colorname,lw)

              % 第一种
             if sqrt((endpoint(1)-startpoint(1))^2+(endpoint(2)-startpoint(2))^2)<=r0
%                  my_draw_arrow1([endpoint(1) endpoint(2)],[startpoint(1) startpoint(2)],colorname, lw)
                 my_draw_arrow1([startpoint(1) startpoint(2)],[endpoint(1) endpoint(2)],colorname, lw)
             end
              % 第二种
              if sqrt((endpoint(1)+D-startpoint(1))^2+(endpoint(2)-startpoint(2))^2)<=r0
                 x01=endpoint(1)+D;y01=endpoint(2);
                 x02=startpoint(1)-D;y02=startpoint(2);
                 y10=(y01-startpoint(2))/(x01-startpoint(1))*(D-x01)+y01;
%                  line([endpoint(1) 0],[endpoint(2) y10],'Color',colorname, lw,'LineWidth',lw);%连线
%                  my_draw_arrow1([D y10],[startpoint(1) startpoint(2)])
                 line([startpoint(1) D],[startpoint(2) y10],'Color',colorname,'LineWidth',lw);%连线
                 my_draw_arrow1([0 y10],[endpoint(1) endpoint(2)],colorname, lw)
              end
              % 第三种
              if sqrt((endpoint(1)-startpoint(1))^2+(endpoint(2)+D-startpoint(2))^2)<=r0
                  x01=endpoint(1);y01=endpoint(2)+D;
                  x02=startpoint(1);y02=startpoint(2)-D;
                  x10=(x01-startpoint(1))/(y01-startpoint(2))*(D-y01)+x01;
%                   line([endpoint(1) x10],[endpoint(2) 0],'Color',colorname,'LineWidth',lw);%连线
%                   my_draw_arrow1([x10 D],[startpoint(1) startpoint(2)])
                  line([startpoint(1) x10],[startpoint(2) D],'Color',colorname,'LineWidth',lw);%连线
                  my_draw_arrow1([x10 0],[endpoint(1) endpoint(2)],colorname, lw)
              end
              % 第四种
              if sqrt((endpoint(1)-D-startpoint(1))^2+(endpoint(2)-startpoint(2))^2)<=r0
                 x01=endpoint(1)-D;y01=endpoint(2);
                 x02=startpoint(1)+D;y02=startpoint(2);
                 y10=(y01-startpoint(2))/(x01-startpoint(1))*(0-x01)+y01;
%                  line([endpoint(1) D],[endpoint(2) y10],'Color',colorname,'LineWidth',lw);%连线
%                  my_draw_arrow1([0 y10],[startpoint(1) startpoint(2)],colorname, lw)
                 line([startpoint(1) 0],[startpoint(2) y10],'Color',colorname,'LineWidth',lw);%连线
                 my_draw_arrow1([D y10],[endpoint(1) endpoint(2)],colorname, lw)
              end
              % 第五种
              if sqrt((endpoint(1)-startpoint(1))^2+(endpoint(2)-D-startpoint(2))^2)<=r0
                  x01=endpoint(1);y01=endpoint(2)-D;
                  x02=startpoint(1);y02=startpoint(2)+D;
                  x10=(x01-startpoint(1))/(y01-startpoint(2))*(0-y01)+x01;
%                   line([endpoint(1) x10],[endpoint(2) D],'Color',colorname,'LineWidth',lw);%连线
%                   my_draw_arrow1([x10 0],[startpoint(1) startpoint(2)],colorname, lw)
                  line([startpoint(1) x10],[startpoint(2) 0],'Color',colorname,'LineWidth',lw);%连线
                  my_draw_arrow1([x10 D],[endpoint(1) endpoint(2)],colorname, lw)
              end
              % 第六种
              if sqrt((endpoint(1)+D-startpoint(1))^2+(endpoint(2)+D-startpoint(2))^2)<=r0
                 x01=endpoint(1)+D;y01=endpoint(2)+D;
                 x02=startpoint(1)-D;y02=startpoint(2)-D;
                 y10=(y01-startpoint(2))/(x01-startpoint(1))*(D-x01)+y01;  % 与x=D 的直线相交于点（D,y10）
                 x10=(x01-startpoint(1))/(y01-startpoint(2))*(D-y01)+x01;  % 与y=D 的直线相交于点（x10,D）
                 if x10<=D
                     x00=x10;y00=D;
                 else
                     x00=D;y00=y10;
                 end
                 y20=(y02-endpoint(2))/(x02-endpoint(1))*(0-x02)+y02;  % 与x=0 的直线相交于点（0,y20）
                 x20=(x02-endpoint(1))/(y02-endpoint(2))*(0-y02)+x02;  % 与y=0 的直线相交于点（x20,0）
                 if x20>=0
                     x11=x20;y11=0;
                 else
                     x11=0;y11=y20;
                 end
%                  line([endpoint(1) x11],[endpoint(2) y11],'Color',colorname,'LineWidth',lw);%连线
%                  my_draw_arrow1([x00 y00],[startpoint(1) startpoint(2)],colorname, lw)
                 line([startpoint(1) x00],[startpoint(2) y00],'Color',colorname,'LineWidth',lw);%连线
                 my_draw_arrow1([x11 y11],[endpoint(1) endpoint(2)],colorname, lw)
              end
              % 第七种
              if sqrt((endpoint(1)-D-startpoint(1))^2+(endpoint(2)-D-startpoint(2))^2)<=r0
                 x01=endpoint(1)-D;y01=endpoint(2)-D;
                 x02=startpoint(1)+D;y02=startpoint(2)+D;
                 y10=(y01-startpoint(2))/(x01-startpoint(1))*(0-x01)+y01;  % 与x=0 的直线相交于点（0,y10）
                 x10=(x01-startpoint(1))/(y01-startpoint(2))*(0-y01)+x01;  % 与y=0 的直线相交于点（x10,0）
                 if x10>=0
                     x00=x10;y00=0;
                 else
                     x00=0;y00=y10;
                 end
                 y20=(y02-endpoint(2))/(x02-endpoint(1))*(D-x02)+y02;  % 与x=0 的直线相交于点（D,y20）
                 x20=(x02-endpoint(1))/(y02-endpoint(2))*(D-y02)+x02;  % 与y=0 的直线相交于点（x20,D）
                 if x20<=D
                     x11=x20;y11=D;
                 else
                     x11=D;y11=y20;
                 end
%                  line([endpoint(1) x11],[endpoint(2) y11],'Color',colorname,'LineWidth',lw);%连线
%                  my_draw_arrow1([x00 y00],[startpoint(1) startpoint(2)],colorname, lw)
                 line([startpoint(1) x00],[startpoint(2) y00],'Color',colorname,'LineWidth',lw);%连线
                 my_draw_arrow1([x11 y11],[endpoint(1) endpoint(2)],colorname, lw)
              end
              % 第八种
              if sqrt((endpoint(1)+D-startpoint(1))^2+(endpoint(2)-D-startpoint(2))^2)<=r0
                 x01=endpoint(1)+D;y01=endpoint(2)-D;
                 x02=startpoint(1)-D;y02=startpoint(2)+D;
                 y10=(y01-startpoint(2))/(x01-startpoint(1))*(D-x01)+y01;  % 与x=D 的直线相交于点（D,y10）
                 x10=(x01-startpoint(1))/(y01-startpoint(2))*(0-y01)+x01;  % 与y=0 的直线相交于点（x10,0）
                 if x10<=D
                     x00=x10;y00=0;
                 else
                     x00=D;y00=y10;
                 end
                 y20=(y02-endpoint(2))/(x02-endpoint(1))*(0-x02)+y02;  % 与x=0 的直线相交于点（0,y20）
                 x20=(x02-endpoint(1))/(y02-endpoint(2))*(D-y02)+x02;  % 与y=D 的直线相交于点（x20,D）
                 if x20>=0
                     x11=x20;y11=D;
                 else
                     x11=0;y11=y20;
                 end
%                  line([endpoint(1) x11],[endpoint(2) y11],'Color','k','LineWidth',lw);%连线
%                  my_draw_arrow1([x00 y00],[startpoint(1) startpoint(2)],colorname, lw)
                 line([startpoint(1) x00],[startpoint(2) y00],'Color',colorname,'LineWidth',lw);%连线
                 my_draw_arrow1([x11 y11],[endpoint(1) endpoint(2)],colorname, lw)
              end
              % 第九种
              if sqrt((endpoint(1)-D-startpoint(1))^2+(endpoint(2)+D-startpoint(2))^2)<=r0
                 x01=endpoint(1)-D;y01=endpoint(2)+D;
                 x02=startpoint(1)+D;y02=startpoint(2)-D;
                 y10=(y01-startpoint(2))/(x01-startpoint(1))*(0-x01)+y01;  % 与x=0 的直线相交于点（0,y10）
                 x10=(x01-startpoint(1))/(y01-startpoint(2))*(D-y01)+x01;  % 与y=D 的直线相交于点（x10,D）
                 if x10>=0
                     x00=x10;y00=D;
                 else
                     x00=0;y00=y10;
                 end
                 y20=(y02-endpoint(2))/(x02-endpoint(1))*(D-x02)+y02;  % 与x=D 的直线相交于点（D,y20）
                 x20=(x02-endpoint(1))/(y02-endpoint(2))*(0-y02)+x02;  % 与y=0 的直线相交于点（x20,0）
                 if x20<=D
                     x11=x20;y11=0;
                 else
                     x11=D;y11=y20;
                 end             
%                  line([endpoint(1) x11],[endpoint(2) y11],'Color','k','LineWidth',lw);%连线
%                  my_draw_arrow1([x00 y00],[startpoint(1) startpoint(2)],colorname, lw)
                 line([startpoint(1) x00],[startpoint(2) y00],'Color',colorname,'LineWidth',lw);%连线
                 my_draw_arrow1([x11 y11],[endpoint(1) endpoint(2)],colorname, lw)
              end

end