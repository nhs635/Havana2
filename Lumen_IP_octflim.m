%%

clc; clear; close all; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OCT FLIM Version
% 
% v161021 drafted
% v161031 updated (MATLAB & C++ Matching: spline factor)
% v161031 updated (Window adjustment method)
% v161114 updated (MATLAB & C++ Matching: new prototype in spline)
% v161130 updated (Software broadening)
% v170209 updated (FLIM mask) 
% v170220 updated (Smart FLIM mask & minor modification)
% v170220 updated (Saturation)
% v170221 updated (minor modification)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% options
full_frame = true; % true : full range frame, false : user-defined range frame
image_view = false;
write = false;

% basic parameters
vis_ch = 2; 

Nsit = 1; 
term = 1; Nit = Nsit;
pp_range = Nsit : term : Nit;

%% INI Parameters

% Filename
filelist = dir;
for i = 1 : length(filelist)
    if (length(filelist(i).name) > 5)
        if strcmp(filelist(i).name(end-4:end),'.data')
            dfilenm = filelist(i).name(1:end-5);
        end
    end
end
clear filelist;

% Please check ini file in the folder
fid = fopen(strcat(dfilenm,'.ini'));
config = textscan(fid,'%s');
fclose(fid); 

ch_start = zeros(1,4);
delay_time_offset = zeros(1,3);
contrast = [70 120];
for i = 1 : length(config{:})
    if (strfind(config{1}{i},'GalvoAdjShift'))
        eq_pos = strfind(config{1}{i},'=');
        galvoshift = str2double(config{1}{i}(eq_pos+1:end));
    end
    
    if (strfind(config{1}{i},'FLIMwidth_factor'))
        eq_pos = strfind(config{1}{i},'=');
        mul = str2double(config{1}{i}(eq_pos+1:end));
    end
    if (strfind(config{1}{i},'FLIMBg'))
        eq_pos = strfind(config{1}{i},'=');
        flim_bg = str2double(config{1}{i}(eq_pos+1:end));
    end
    if (strfind(config{1}{i},'FLIMCh'))
        eq_pos = strfind(config{1}{i},'=');
        flim_ch = str2double(config{1}{i}(eq_pos+1:end)); 
        % 0 -> 4n / 1 -> 4n+1 / 2 -> 4n+2 / 3 -> 4n+3
    end
    if (strfind(config{1}{i},'PreTrigSamps'))
        eq_pos = strfind(config{1}{i},'=');
        pre_trig = str2double(config{1}{i}(eq_pos+1:end));
    end
    if (strfind(config{1}{i},'nAlines'))
        eq_pos = strfind(config{1}{i},'=');
        n_alines = str2double(config{1}{i}(eq_pos+1:end));
    end
    if (strfind(config{1}{i},'Max_dB'))
        eq_pos = strfind(config{1}{i},'=');
        contrast(2) = str2double(config{1}{i}(eq_pos+1:end));
    end
    if (strfind(config{1}{i},'Min_dB'))
        eq_pos = strfind(config{1}{i},'=');
        contrast(1) = str2double(config{1}{i}(eq_pos+1:end));
    end
    for j = 0 : 3
        ChannelStart = sprintf('ChannelStart_%d',j);
        if (strfind(config{1}{i},ChannelStart))
            eq_pos = strfind(config{1}{i},'=');
            ch_start(j+1) = str2double(config{1}{i}(eq_pos+1:end)); 
        end
    end
    for j = 1 : 3
        DelayTimeOffset = sprintf('DelayTimeOffset_%d',j);
        if (strfind(config{1}{i},DelayTimeOffset))
            eq_pos = strfind(config{1}{i},'=');
            delay_time_offset(j) = str2double(config{1}{i}(eq_pos+1:end)); 
        end
    end    
end

% Size parameters
n4_alines = n_alines/4;
n_scans = 2600;
n2_scans = n_scans/2; nn_scans = n_scans*2; fn_scans = n_scans*4;
n_len = 2^ceil(log2(n_scans)); nn_len = n_len*2;
n2_len = n_len/2; n4_len = n_len/4; n8_len = n_len/8;

im_size = n_len*n_alines;
im2_size = n2_len*n_alines;

samp_intv = 1000/340;
ch_start_ind = round(ch_start./samp_intv)+1;
ch_start_ind0 = ch_start_ind - ch_start_ind(1) + 1;
ch_start_ind0 = [ch_start_ind0 ch_start_ind0(end) + 30];

if (write), mkdir('rect_image'); end

clear config ChannelStart DelayTimeOffset eq_pos;

%% background frame

bgname = [dfilenm,'.background'];
fid = fopen(bgname,'r','l');
oct_bg = fread(fid,'uint16');
fclose(fid);

bgmm = reshape(oct_bg(1:n_scans*n_alines),n_scans,n_alines); % 2600 * 1024
% bgm1 = mean(bgmm(:,1:2:end), 2);
% bgm2 = mean(bgmm(:,2:2:end), 2);
% bgm = [bgm1 bgm2];
% bg = repmat(bgm, 1, n_alines / 2);
bgm = mean(bgmm, 2);
bg = repmat(bgm, 1, n_alines);

clear fid_bx fid_by data_b oct_bg bgmm bgm;

%% calibration data

fid_cal = fopen([dfilenm,'.calibration'], 'r');
fseek(fid_cal, 0, 'bof');
int_ind = fread(fid_cal, n2_scans, 'float32') + 1;
interp11 = int_ind; %(n2_scans:-1:1);
interp22 = interp11+1;

fseek(fid_cal, 0, 'cof');
weight_c = fread(fid_cal, n2_scans, 'float32');
weight1 = weight_c; %(n2_scans:-1:1);
weight2 = 1-weight1;

fseek(fid_cal, 0, 'cof');
dspr = fread(fid_cal, n2_scans, 'float32');
dsp_real = dspr; %(n2_scans:-1:1);

fseek(fid_cal, 0, 'cof');
dspi = fread(fid_cal, n2_scans, 'float32');
dsp_imag = dspi; %(n2_scans:-1:1);

fclose(fid_cal);
clear int_ind weight_c dspr dspi;

window = hann(n_scans);

discomidex=transpose(1:1:n2_scans);
discom_parameter=0;
discom=exp(1i*discom_parameter*((discomidex-n2_scans/2)/n2_scans).^2);

dsp = (dsp_real + 1i*dsp_imag) .* discom;
dsp1 = repmat(dsp,[1 n_alines]);

clear discomidex dsp discom;

%% parameters for FLIM operation

% FLIM mask
fid_mask = fopen([dfilenm,'.flim_mask'], 'r');
FLIM_mask = fread(fid_mask,'float32');
fclose(fid_mask);

start_ind = find(diff(~FLIM_mask) == 1) + 1;
end_ind   = find(diff(~FLIM_mask) == -1) - 2;

is_mask   = sum(FLIM_mask-1) ~= 0;

% saturation indicator
max_val_flim = 65532 - flim_bg;

% range data
roi_range = ch_start_ind(1) + pre_trig : ch_start_ind(4) + 30 + pre_trig - 1;
scale_factor = 20;
actual_factor = (length(roi_range)*scale_factor-1)/(roi_range(end)-roi_range(1));

roi_range1 = linspace(roi_range(1),roi_range(end),length(roi_range)*scale_factor);    

[xxold,yyold] = meshgrid(1:n4_alines,roi_range);
[xxnew,yynew] = meshgrid(1:n4_alines,roi_range1); 

ch_start_ind1 = round(ch_start_ind * actual_factor);

hg = fspecial('Gaussian',[scale_factor*10 1],scale_factor*2.4);

min_interval = min([diff(ch_start_ind1) 30*actual_factor]);
range_pulse = 1 : min_interval;

%% OCT FLIM data

fname11 = sprintf([dfilenm,'.data']);
fid_dx = fopen(fname11,'r','l');
fseek(fid_dx,0,'eof');
n_frame = ftell(fid_dx)/2/nn_scans/n_alines;

if (full_frame)
    pp_range = 1 : n_frame;
end

for pp = pp_range
    clc; pp 
  
    % load data
    fid_dx = fopen(fname11,'r','l');
    fseek(fid_dx, (pp-1)*2*nn_scans*n_alines, 'bof');
    ft = fread(fid_dx,nn_scans*n_alines,'uint16=>single');  
        
    % Process OCT data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Deinterleave data
    f_oct  = reshape(ft(1:2:nn_scans*n_alines),n_scans,n_alines);
            
    % BG removal & Windowing & Zero padding
    data_signal = [(f_oct - bg) .* repmat(window,[1 n_alines]); zeros(n_len-n_scans,n_alines)];
    
    % FFT real data
    dft_signal = fft(data_signal);
    clear f_oct data_signal;
    
    % Copy to extended vector, zero padding
    dft_signal1 = circshift(dft_signal,+n4_len); % frequencey shifting compensation
    dft_signal_zoom1 = [dft_signal1(1:n4_len,:); ...
        zeros(n2_len,n_alines); dft_signal1(3*n4_len+1:4*n4_len,:);]; % mirror image removal    
    clear dft_signal dft_signal1;
    
    % IFFT to get complex fringes
    scan_signal_zoom1 = ifft(dft_signal_zoom1);    
    clear dft_signal_zoom1;
    
    dft_complex1_re = zeros(n2_scans, n_alines);
    dft_complex1_im = zeros(n2_scans, n_alines);    
    
    % Calibration
    for ii = 1 : n2_scans
        dft_complex1_re(ii,:) = (weight1(ii)*real(scan_signal_zoom1(interp11(ii),:)) ...
                                        + weight2(ii)*real(scan_signal_zoom1(interp22(ii),:)));
        dft_complex1_im(ii,:) = (weight1(ii)*imag(scan_signal_zoom1(interp11(ii),:)) ...
                                        + weight2(ii)*imag(scan_signal_zoom1(interp22(ii),:)));
    end                 
    dft_complex1 = dft_complex1_re + 1i*dft_complex1_im;        
    dft_complex2 = dsp1.*dft_complex1;    
    
    dft_complex_zoom1 = fft(dft_complex2, n2_len);               
    Itt1 = circshift(abs(dft_complex_zoom1).^2, n4_len);     
    
    clear scan_signal_zoom1 dft_complex* dft_complex_zoom1;    
    
    % Scaling 8 bit image
    xyim1 = 255 * mat2gray(10*log10(Itt1),contrast);
    xyim1 = circshift(xyim1',-galvoshift)';
    xyim1 = medfilt2(xyim1,[3 3]);
    xyim2 = 255 - xyim1;
    
    if (write)
        bmpname = sprintf(['rec_1',dfilenm,'_%03d.bmp'],pp);
        imwrite(uint8(xyim2), strcat('./rect_image/',bmpname), 'bmp');    
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Process FLIM data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    % Deinterleave data
    f_flim = double(reshape(ft(2:2:nn_scans*n_alines),n_scans,n_alines));
    f_flim1 = f_flim(roi_range,(flim_ch+1):4:n_alines) - flim_bg;
    
    % Saturation detection
    is_saturation = zeros(4,n4_alines);
    for ch = 1 : 4
        for ii = 1 : n4_alines
            is_saturation(ch,ii) = ...
                length(find(f_flim1(ch_start_ind0(ch):ch_start_ind0(ch+1)-1,ii) == max_val_flim));
        end
    end
    
    % Masking for removal of RJ artifact (if necessary)
    if (is_mask)
        for ii = 1 : n4_alines
            for i = 1 : 4
                win_range = start_ind(i):end_ind(i);
                if (i == 1)
                    f_flim1(win_range,ii) = 0;
                else
                    [~,max_ind] = max(f_flim1(win_range,ii));
                    max_ind = max_ind + win_range(1) - 1;
                    win_range1 = max_ind - 4:max_ind + 3;
                    f_flim1(win_range1,ii) = 0;
                end
            end  
        end
    end
    
    % Spline interpolation   
    f_flim_ext = interp2(xxold,yyold,f_flim1,xxnew,yynew,'spline');   
    
    % Software broadening
    f_flim_ext1 = zeros(size(f_flim_ext));
    for ii = 1 : n4_alines
        f_flim_ext1(:,ii) = filter(hg,1,f_flim_ext(:,ii));
    end

    % Fluorescence extraction    
    flu_intensity  = zeros(4, n4_alines);
    flu_mean_delay = zeros(4, n4_alines);
    flu_lifetime   = zeros(3, n4_alines);    

    for ii = 1 : n4_alines   

        data0 = f_flim_ext (:,ii);
        data1 = f_flim_ext1(:,ii);   

        % Find mean delay
        for jj = 1 : 4
            offset = (ch_start_ind1(jj) - ch_start_ind1(1));
            range_pulse1 = range_pulse + offset;
            data2 = data0(range_pulse1);
            
            % Find intensity
            flu_intensity(jj,ii) = sum(data2);
            
            if (sum(data2) > 0.001)
                % Find width of the pulse (adjustable window method)
                r = 0; l = 0;
                data2 = data1(range_pulse1);
                [peakval,peakind] = max(data2);            
                for k = peakind : min_interval
                    if (data2(k) < peakval * 0.5), r = k; break; end
                end
                for k = peakind : -1 : 1
                    if (data2(k) < peakval * 0.5), l = k; break; end
                end

                irf_width = mul * (r - l + 1);
                left = floor((irf_width - 1)/2);    
                md_new = peakind;

                % Find mean delay
                for iter = 1 : 5 
                    start = round(md_new) - left;
                    range_pulse2 = start : start + irf_width - 1;            
                    range_pulse2 = range_pulse2(range_pulse2<=length(data2)&(range_pulse2>0));
                    md_new = (range_pulse2) * data2(range_pulse2) / sum(data2(range_pulse2));
                    if isempty(md_new), md_new = 1; break; end  
                    if isnan(md_new), md_new = 1; break; end 
                end

                md_new1 = md_new + offset + (pre_trig + ch_start_ind(1)) * actual_factor;
                flu_mean_delay(jj,ii) = (md_new1 - 1) * samp_intv/actual_factor;

                % Find lifetime
                if (jj ~= 1)  
                    flu_lifetime(jj-1,ii) = (flu_mean_delay(jj,ii) - flu_mean_delay(1,ii)) - delay_time_offset(jj-1);
                end      
            else
                if (jj ~= 1)  
                    flu_lifetime(jj-1,ii) = 0;
                end   
            end
        end
        
        % Find normalized intensity
        flu_intensity(:,ii) = flu_intensity(:,ii)./flu_intensity(1,ii);   
    end
    
    flu_intensity = circshift(flu_intensity', -round(galvoshift/4))';
    flu_lifetime  = circshift(flu_lifetime' , -round(galvoshift/4))';      
    
    intensity_map(:,:,pp) = flu_intensity;
    lifetime_map(:,:,pp) = flu_lifetime;
    meandelay_map(:,:,pp) = flu_mean_delay; 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
       
    if (image_view)        
        intensity1 = medfilt1(flu_intensity(vis_ch+1,:),5);
        lifetime1  = medfilt1(flu_lifetime (vis_ch,:),  1);
                
        figure(1); set(gcf,'Position',[420 180 516 778]);
        imshow(uint8(xyim2)); colormap gray; freezeColors; hold on;
        h1 = imshow(imresize(repmat(lifetime1,[50 1]),[50 n_alines])); 
        caxis([0 10]); colormap jet; freezeColors;
        h1.YData = [2048-50 2048];
        h2 = imshow(imresize(repmat(intensity1,[50 1]),[50 n_alines])); 
        caxis([0 2]); colormap hot; freezeColors; hold off;
        h2.YData = [2048-120 2048-70];

        figure(2); set(gcf,'Position',[952 538 560 420]);
        subplot(211); plot(intensity1); hold on;
        plot(find(is_saturation(vis_ch+1,:)~=0),intensity1(is_saturation(vis_ch+1,:)~=0),'ro'); 
        plot(find(is_saturation(1,:)~=0),       intensity1(is_saturation(1,:)~=0),       'ko'); hold off;
        axis([0 n4_alines 0 2]); ylabel('normalized intensity');
        title(sprintf('normalized intensity (hot): %.4f +- %.4f AU',mean(intensity1), std(intensity1))); 
               
        subplot(212); plot(lifetime1); hold on;
        plot(find(is_saturation(vis_ch+1,:)~=0),lifetime1(is_saturation(vis_ch+1,:)~=0),'ro'); 
        plot(find(is_saturation(1,:)~=0),lifetime1(is_saturation(1,:)~=0),'ko'); hold off;
        axis([0 n4_alines 0 10]); ylabel('lifetime'); 
        title(sprintf('lifetime (jet): %.4f +- %.4f nsec', mean(lifetime1), std(lifetime1)));               
    end           
    
    fclose('all');    
end

fclose('all');

%%

if (full_frame)
    ch = 2;

    i_map = reshape(intensity_map(ch+1,:,:),[n4_alines n_frame]); 
    t_map = reshape(lifetime_map(ch,:,:),[n4_alines n_frame]); 

    i_map = medfilt2(i_map,[5 3],'symmetric');
    t_map = t_map .* (i_map > 0.001);
    t_map = medfilt2(t_map,[5 3],'symmetric');

    figure(1);
    subplot(121); imagesc(circshift(i_map,-round(galvoshift/4))); 
    caxis([0 0.5]); colormap hot; freezeColors;
    subplot(122); imagesc(circshift(t_map,-round(galvoshift/4))); 
    caxis([0 5]); colormap jet; freezeColors;
    
    if (write)        
        mkdir('FLIMres');
        
        itn_name = 'FLIMres/intensity_ch%d.flimres';
        lft_name = 'FLIMres/lifetime_ch%d.flimres';
        
        for i = 1 : 3
            fid_itn = fopen(sprintf(itn_name,i),'wb');
            fid_lft = fopen(sprintf(lft_name,i),'wb');
            
            i_map = intensity_map(i+1,:,:);
            t_map = lifetime_map(i,:,:);
            
            fwrite(fid_itn,i_map(:),'float');
            fwrite(fid_lft,t_map(:),'float');
            
            fclose(fid_itn);
            fclose(fid_lft);
        end
    end
    
end
