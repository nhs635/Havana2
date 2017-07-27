clear;

fid = fopen('fire.colormap','r');
fire = fread(fid,'uint8'); 
fire = reshape(fire,[256 3])/255;
fclose(fid);

hsv1 = hsv(256);
hsv1(1,:) = 0;

%%

filelist = dir('*.bmp');
mkdir('scale_indicated');

for i = 1 : length(filelist)
    filename = filelist(i).name;
    im = uint8(imread(filename));
    
    figure(2); set(gcf,'Position',[455   379   650   445]);
    imagesc(im); caxis([0 255]);
    xlabel('angle'); ylabel('frame');
    
    if (filename(1) == 'i')
        colormap(fire);
        s = find(filename == '[');
        e = find(filename == ']');
        contrast = [ str2double(filename(s+1:s+3)) str2double(filename(e-3:e-1)) ];
    elseif (filename(1) == 'l')
        colormap(hsv1);  
        s = find(filename == '[');
        e = find(filename == ']');
        contrast = [ str2double(filename(s+1:s+3)) str2double(filename(e-3:e-1)) ];
    elseif (filename(1) == 'f')
        colormap(hsv1);
        s = find(filename == '[');
        e = find(filename == ']');
        contrast = [ str2double(filename(s(2)+1:s(2)+3)) str2double(filename(e(2)-3:e(2)-1)) ];
    elseif (filename(1) == 'o')
        colormap(gray);
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
