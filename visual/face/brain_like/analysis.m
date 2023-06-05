%% 预处理
clear;
close all;
stimuli_num = 2000;
% 包含 stimuli 文件夹中的数据
if ~exist("Stimuli", "dir"); mkdir("Stimuli"); end
% response_am.mat, response_mlmf.mat
if ~exist("response", "dir"); mkdir("response"); end
% 包含 generate_face_50d_space 文件夹中的文件
if ~exist("feature", "dir"); mkdir("feature"); end

%% 加载数据
load('response\response_am.mat');  % 加载前内(AM)的响应，保存到变量response里
% response_am = response;
% load features
for k=1:stimuli_num
    fid=fopen(['.\stimuli\',num2str(k),'.txt'],'r');
    ao=fscanf(fid,'%f ');
    para(k,:)=ao;
    fclose(fid);
end

for j=1:size(para,2)
    para(:,j)=para(:,j)-mean(para(:,j));
    lo(j)=std(para(:,j),1);
end
lo1=(sum(lo(1:25).*lo(1:25)))^0.5;
lo2=(sum(lo(26:50).*lo(26:50)))^0.5;
para(:,1:25)=para(:,1:25)/lo1;
para(:,26:50)=para(:,26:50)/lo2;
para=para/sqrt(2);
%compute STA for a single cell
fir=response(1,:);
sta(1:50)=0;
for kj=1:stimuli_num
    sta=sta+para(kj,1:50)*fir(kj);
end
sta=sta/sum(fir);

% 论文中的图1.D
figure;
subplot(1,2,1);
plot(1:25,sta(1:25),'MarkerFaceColor',[0 0 0],'Marker','o','Color',[0 0 0]);
hold('all');plot(26:50,sta(26:50),'MarkerFaceColor',[0 0 0],'Marker','o','Color',[0 0 0]);
xlim([0 51]);
ylabel('STA');
xlabel('Feature Dimension');

%compute tuning along STA axis
stan=sta./norm(sta);
for i=1:stimuli_num
    generator(i)=sum(para(i,:).*sta(1:50));
end
cei=max(abs(generator))*1.01;
gen=[];num=[];
rule=20;
gen(1:rule)=0;
num(1:rule)=0;
gen2(1:rule,1:400)=0;
for i=1:size(para,1)
    scale0=generator(i);
    sca=floor(rule*(scale0+cei)/(2*cei))+1;
    gen(sca)=gen(sca)+fir(i);
    num(sca)=num(sca)+1;
    gen2(sca,num(sca))=fir(i);
end

me=[];
st=[];
me(1:rule)=0;
st(1:rule)=0;
for i=1:rule
    me(i)=mean(gen2(i,1:num(i)),2);
    if num(i)>1
        st(i)=std(gen2(i,1:num(i)),0,2)/sqrt(num(i));
    else
        st(i)=0;
        me(i)=1/0;
    end
end
xx=-cei+cei/rule:2*cei/rule:cei-cei/rule;

% 论文中的图1.I
subplot(1,2,2);
errorbar(xx,me,st,'MarkerFaceColor',[0 0 0],'Marker','o','Color',[0 0 0]);
ylabel('Firing rate (spikes/s)');
xlabel('Distance along STA axis');

%compare feature prefrence across two areas
response_am=response;
load('response\response_mlmf.mat');  % 加载 ML/MF 的神经响应
response_mlmf=response;

for i=1:size(response_am,1)
    fir=response_am(i,:);
    sta(1:50)=0;
    for kj=1:stimuli_num
        sta=sta+para(kj,1:50)*fir(kj);
    end
    sta=sta/sum(fir);
    fpam(i)=(norm(sta(1:25))-norm(sta(26:50)))/(norm(sta(1:25))+norm(sta(26:50)));
end

for i=1:size(response_mlmf,1)
    fir=response_mlmf(i,:);
    sta(1:50)=0;
    for kj=1:stimuli_num
        sta=sta+para(kj,1:50)*fir(kj);
    end
    sta=sta/sum(fir);
    fpmlmf(i)=(norm(sta(1:25))-norm(sta(26:50)))/(norm(sta(1:25))+norm(sta(26:50)));
end

rule=40;
sp(1:rule+1)=0;
for kl=1:length(fpmlmf)
    sc0=round((fpmlmf(kl)+1)*rule/2)+1;
    sp(sc0)=sp(sc0)+1;
end
xp=-1:1/20:1;
spp(1:2:81)=sp;
spp(2:2:80)=sp(1:end-1);
xpp(1:2:81)=xp;
xpp(2:2:80)=xp(2:end);

sp2(1:rule+1)=0;
for kl=1:length(fpam)
    sc0=round((fpam(kl)+1)*rule/2)+1;
    sp2(sc0)=sp2(sc0)+1;
end
spp2(1:2:81)=sp2;
spp2(2:2:80)=sp2(1:end-1);

figure;plot(xpp,spp);hold('all');plot(xpp,spp2);
ylabel('Number of Cells');
xlabel('(S-A)/(S+A)');
legend('MLMF','AM')

%decode_features_from_population_responses
response_all=[response_mlmf;response_am];
for pa=1:50
    for id=1:stimuli_num
        if id==1
            z=2:stimuli_num;
        else
            z=[1:id-1,id+1:stimuli_num];
        end
        g2=[response_all(:,z)]';
        g2(:,size(g2,2)+1)=1;
        [b,bint,r,rint,stats] = regress(para(z,pa),g2);
        if id==1
            [a, MSGID] = lastwarn();
            warning('off', MSGID);
        end
        y2=g2*b;
        we(id)=[response_all(:,id);1]'*b;
    end
    for i=1:length(we)
        if ~(we(i)<5000000)
            we(i)=0;
        end
    end
    lk1=we;
    lk2=para(1:length(we),pa)';
    ev=1-sum((lk1-lk2).*(lk1-lk2))/sum((lk2-mean(lk2)).*(lk2-mean(lk2)));
    evtt(pa)=ev;
    prepara(1:id,pa)=lk1(1:id);
end

figure;subplot(1,2,1);
scatter(para(:,26),prepara(:,26),'MarkerFaceColor',[0 0 0],'MarkerEdgeColor',[0 0 0]); xlim([-1.25 1.25]);ylim([-1.25 1.25]);hold('all');plot([-1.25 1.25],[-1.25 1.25],'color',[0 0 0]);
ylabel('Predicted Parameter ');
xlabel('Actual Parameter (first appearance dimension)');

subplot(1,2,2);
plot(1:25,evtt(1:25),'MarkerFaceColor',[0 0 0],'Marker','o','Color',[0 0 0]);
hold('all');plot(26:50,evtt(26:50),'MarkerFaceColor',[0 0 0],'Marker','o','Color',[0 0 0]);
xlim([0 51]);
ylabel('Explained Variance (%)');
xlabel('Feature Dimension');