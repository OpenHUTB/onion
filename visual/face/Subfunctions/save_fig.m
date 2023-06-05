% 被 export_fig 工具箱代替
function save_fig(h, filename)  
% 将当前绘制的图像保存到latex的figs目录下（以pdf的格式），
% 参考：https://baijiahao.baidu.com/s?id=1704887660228490516&wfr=spider&for=pc

% h = figure;%% -----程序中的figure；plot(1:10);
% 自动提取图片的尺寸和范围；
% set(gcf,'Units','inches');
% screenposition = get(gcf,'Position');
% set(gcf,...    
%     'PaperPosition',[0 0 screenposition(3:4)],...    
%     'PaperSize',[screenposition(3:4)]);
% save figure as pdfprint -dpdf -painters figure1

set(h, 'Units', 'Inches');
pos = get(h, 'Position');
set(h, 'PaperPositionMode','Auto', ...
    'PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

set(gca, 'LooseInset', [0,0,0,0]);


% 保存图片为pdf格式，分辨率设置为600
print(h, filename, '-dpdf','-r600')
close(h)

end

