clear;

filelist = dir('*.bmp');
mkdir('scale_indicated');

for i = length(filelist) : -1 : 1
    filename = filelist(i).name;
    [im,map] = imread(filename);
    
    figure(2); set(gcf,'Position',[455   379   650   445]);
    imagesc(im); caxis([0 255]);
    xlabel('angle'); ylabel('frame');
    
    if (filename(1) == 'i')
        intensity_map = map;
        colormap(intensity_map);
        s = find(filename == '[');
        e = find(filename == ']');
        contrast = [ str2double(filename(s+1:s+3)) str2double(filename(e-3:e-1)) ];
    elseif (filename(1) == 'l')
        lifetime_map = map;
        colormap(lifetime_map);  
        s = find(filename == '[');
        e = find(filename == ']');
        contrast = [ str2double(filename(s+1:s+3)) str2double(filename(e-3:e-1)) ];
    elseif (filename(1) == 'f')
        colormap(lifetime_map);
        s = find(filename == '[');
        e = find(filename == ']');
        contrast = [ str2double(filename(s(2)+1:s(2)+3)) str2double(filename(e(2)-3:e(2)-1)) ];
    elseif (filename(1) == 'o')
        oct_map = map;
        colormap(oct_map);
        contrast = [ 0 0 ];
    end
        
    h = colorbar;       
    set(h,'Ticks',[0 255]);
    set(h,'TickLabels',cellstr(num2str(contrast')));    
       
    dot_pos = find(filename == '.');
    filename2 = filename(1:dot_pos(end)-1);
   
    filename3 = filename2;
    filename3(filename3 == '_') = ' ';
    title(filename3);
    
    im1 = frame2im(getframe(gcf));
    imwrite(im1,strcat('scale_indicated/',filename2,'_scale.bmp')); 
end
