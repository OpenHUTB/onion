function pingpong
% written by Lindo Ouseph
% Research scholar, Department of Electronics,CUSAT,Kochi,India
% PinPong game.Enjoy
%use uparrow and downarrow to control the bat
global Vx Vy w Step count h D
close all;clc
Vx=.001;Vy=.002;w=600;Step=0.001;count=0;D=1;
s=get(0,'screensize');
h.f=figure('menubar','none','numbertitle','off','name','PingPong','position',[s(3)/2-300,s(4)/2-300,w,w]);
h.a=axes('xlim',[0 1],'ylim',[0 1],'xtick',[],'ytick',[],'box','on','position',[0 0 1 1]);axis square
h.p=patch([.99 .96 .96 .99],[.3 .3 .4 .4],[1 0 0 ]);
h.b=patch(.5+.02*sin([0:.01:1]*2*pi),.5+.02*cos([0:.01:1]*2*pi),[0 0 1]);
h.timer=timer('TimerFcn',@Beat,'ExecutionMode','fixedDelay','Period',.001);
h.t=text(0.01,.97,'hi','color',[.5 .8 0],'fontweight','bold','fontsize',20);
set(h.f,'keypressFcn',{@Down,h});
start(h.timer);
%--------------------------------------------------------------------------
function Down(~,evnt,h)
global Step D
if strcmpi(evnt.Key,'downarrow')
    if D
        Step=-0.001;
        D=~D;
    else
        Step=Step-0.0005;
    end
elseif strcmpi(evnt.Key,'uparrow')
    if D
        Step=Step+0.0005;
    else
        Step=0.001;
        D=~D;
    end
else
    return;
end
set(h.p,'ydata',get(h.p,'ydata')+Step);
drawnow;
%--------------------------------------------------------------------------
function Beat(varargin)
global Vx Vy w Step count h D
Bx=unique(get(h.b,'xdata'));Bx=Bx([1,end]);
By=unique(get(h.b,'ydata'));By=By([1,end]);
Rx=unique(get(h.p,'xdata'));Rx=Rx([1,end]);
Ry=unique(get(h.p,'ydata'));Ry=Ry([1,end]);

if (Bx(2)>=Rx(1) && By(2)>=Ry(1) && By(1)<=Ry(2))
    Vx=-0.001;
    Vy=Vy+Step;
    count=count+1;
    set(h.t,'string',num2str(count));drawnow
end
if By(1)<=0
    Vy=0.001;
    D=~D;
elseif By(2)>=1
    Vy=-0.001;
    D=~D;
end
if Bx(1)<=0
    Vx=0.001;
elseif Bx(2)>=1
    set(h.t,'string','Game Over !');drawnow
    stop(h.timer);
end
Py=unique(get(h.p,'ydata'));Py=Py([1,end]);
if Py(1)<=0
    Step=0.001;
elseif Py(2)>=1
    Step=-0.001;
end
set(h.b,'xdata',get(h.b,'xdata')+Vx);
set(h.b,'ydata',get(h.b,'ydata')+Vy);
set(h.p,'ydata',get(h.p,'ydata')+Step);
drawnow;
%--------------------------------------------------------------------------