function export_figure(h, filename)

set(h,'Units','Inches');
pos = get(h,'Position');
set(gcf, 'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize', [pos(3), pos(4)])
print(gcf, filename, '-dpdf','-r0')
close(h)

end