clear
close all

defaultpath=pwd;
subName =strtrim( input('subject name is ', 's')); % 被験者名の入力
[folderPath] = uigetdir();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mkdir([defaultpath '\data\Trainingdata']);
addpath(genpath([defaultpath '\functions']))

NoImage=100;
RecDuration=4;%sec ; time limit of 1 trial %
numrep=1;
ThuZ=2;
%%sub
savefilename=[defaultpath '\data\Trainingdata\' subName  '-' datestr(now,30)];

%% imagelist
imgpath=[folderPath '\iter_0'];
IMGlist=dir([imgpath '\*.jpg']);
imglist={IMGlist.name};

Question={'好き'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXperiment Setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
AssertOpenGL;
%% Keyboardのチェック
KbName('UnifyKeyNames');  % OSで共通のキー配置にする
DisableKeysForKbCheck([]);% 無効にするキーの初期化
[ keyIsDown, secs, keyCode ] = KbCheck;% 常に押されるキー情報を取得する
keys=find(keyCode) ;%
DisableKeysForKbCheck(keys);

% %% Variablesi
Screen('Preference', 'SkipSyncTests', 1);%　これを入れないと時々エラーが出る
Screen('Preference', 'TextRenderer', 1);
Screen('Preference', 'TextAntiAliasing', 0);
Screen('Preference', 'TextAlphaBlending', 0);
windowsize=[];
screenid = max(Screen('Screens'));
ScreenDevices = java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment().getScreenDevices();
MainScreen = java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice().getScreen()+1;
MainBounds = ScreenDevices(MainScreen).getDefaultConfiguration().getBounds();
MonitorPositions = zeros(numel(ScreenDevices),4);
for n = 1:numel(ScreenDevices)
    Bounds = ScreenDevices(n).getDefaultConfiguration().getBounds();
    MonitorPositions(n,:) = [Bounds.getLocation().getX() + 1,-Bounds.getLocation().getY() + 1 - Bounds.getHeight() + MainBounds.getHeight(),Bounds.getWidth(),Bounds.getHeight()];
end

% [w, rect] = Screen('OpenWindow', screenid, 128, [10 30 400 200]); %test用
[w, rect] = Screen('OpenWindow',screenid, 128 , [MonitorPositions(1,1)  MonitorPositions(1,2) MonitorPositions(1,1)+MonitorPositions(1,3) round(MonitorPositions(1,4)*0.75)]); %test

Screen('TextFont',w, 'MS Mincho');
Screen('TextSize',w, 25);
Screen('TextStyle', w, 0);
Screen('BlendFunction', w, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
Screen('TextFont', w, '-:lang=ja');

windowsize=get(0,'MonitorPositions');
h=figure('Position',[windowsize(1,1) windowsize(1,2) windowsize(1,3) 230],'Color','k');%alwaysontop(h)
h.MenuBar='none';h.ToolBar='none';
[centerX, centerY] = RectCenter(rect);%画面の中央の座標
[screenXpixels, screenYpixels] = Screen('WindowSize', w);
% imgサイズを計算 (画面の高さの 1/2)
sideLength = screenYpixels / 2;
baseRect = [0, 0, sideLength, sideLength];
halfSide = sideLength / 2;
left   = centerX - halfSide;
top    = centerY - halfSide;
right  = centerX + halfSide;
bottom = centerY + halfSide;

% 4. 描画用の矩形を作成
destRect = [left, top, right, bottom];

HideCursor();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Stim setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stimPath = IMGlist(1).folder;  % 適宜変更

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EEG SETTING (VIE ZONE)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
selpath=['C:\VIErecordingFolder'];
filelist=dir([selpath '\VieOutput\VieRawData_*.csv']);
for i=1:length(filelist);delete([filelist(i).folder '\' filelist(i).name]);end
filename=[selpath '\VieOutput\' 'VieRawData.csv' ];%
opts = detectImportOptions(filename);
opts.SelectedVariableNames = [2 3];
SAMPLE_FREQ_VIE=600;
FreqWindow=[4:0.5:40];
timeWindow=4;

%% Filter Setting
[B1f,A1f] = butter(4,[3/(SAMPLE_FREQ_VIE/2) 40/(SAMPLE_FREQ_VIE/2)]);% 3~40Hzバタワースフィルタ設計
Zf1=[];

%% EEG Monitor Window
mtimeWindow=4;
cplotdata=repmat(0,SAMPLE_FREQ_VIE*mtimeWindow,3)+[40 0 -40];
cidx=1;
cbaseline=[0 0 0];

% % Signal Quality Parameter
NoiseMetricsLabel={'Average Power_L' 'Average Power_R' 'RMS_L' 'RMS_R' 'MaxGradient_L' 'MaxGradient_R' 'ZeroCrossing Rate_L' 'ZeroCrossing Rate_R' 'Kurtosis_L' 'Kurtosis_R'};
NoiseMU=[138.502300000000	138.502300000000	12.1827000000000	12.1827000000000	103.387000000000	103.387000000000	0.0353000000000000	0.0353000000000000	10.1240000000000	10.1240000000000];
NoiseSigma=[123.240600000000	123.240600000000	6.02540000000000	6.02540000000000	69.4416000000000	69.4416000000000	0.00390000000000000	0.00390000000000000	2.89900000000000	2.89900000000000];

%Screen setting
ScreenDevices = java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment().getScreenDevices();
MainScreen = java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice().getScreen()+1;
MainBounds = ScreenDevices(MainScreen).getDefaultConfiguration().getBounds();
MonitorPositions = zeros(numel(ScreenDevices),4);
for n = 1:numel(ScreenDevices)
    Bounds = ScreenDevices(n).getDefaultConfiguration().getBounds();
    MonitorPositions(n,:) = [Bounds.getLocation().getX() + 1,-Bounds.getLocation().getY() + 1 - Bounds.getHeight() + MainBounds.getHeight(),Bounds.getWidth(),Bounds.getHeight()];
end

windowsize=get(0,'MonitorPositions');
h=figure('Position',[windowsize(1,1) windowsize(1,2) windowsize(1,3) round(windowsize(1,4)*0.22)],'Color','k');%alwaysontop(h)
h.MenuBar='none';h.ToolBar='none';

EEGplot=plot([1:SAMPLE_FREQ_VIE*mtimeWindow]/SAMPLE_FREQ_VIE,cplotdata-cbaseline,'LineWidth',1);
ha1 = gca;ha1.GridColor=[1 1 1];
set(gca,'Color','k')
h_yaxis = ha1.YAxis; % 両 Y 軸の NumericRulerオブジェクト(2x1)を取得
h_yaxis.Color = 'w'; % 軸の色を黒に変更
h_yaxis.Label.Color = [1 1 1]; %  軸ラベルの色変更
h_xaxis = ha1.XAxis; % 両 Y 軸の NumericRulerオブジェクト(2x1)を取得
h_xaxis.Color = 'w'; % 軸の色を黒に変更
h_xaxis.Label.Color = [1 1 1]; %  軸ラベルの色変更
set(gca,'Color','k')
xlim([0 4]);
titletext=title(['EEG (' num2str(round(0,2)) 's)'],'Color' ,'w', 'FontSize', 22);
xlabel('time (s)');
ylabel('uV');
yline(0,'w--');
ylim([-75 75]);
lgd=legend(EEGplot,{'L' 'R' 'diff'});
lgd.TextColor=[1 1 1];

%% Scale Setting
LineHaba = 7; %線分の幅
Linelength = rect(3)/2;
LineVertPos = rect(4)+10; %水平線分の垂直方向の位置
LineColor = [255 255 255];
AgeHeight=280;%テキスト表示位置
AdjustPar=100;%調整用;

minx= centerX-Linelength/2;
maxx= centerX+Linelength/2;

EndFlag=0;

%% ready
DrawFormattedText(w,double(['画像の好みを回答してください。\n波形がきれいになったらクリックして回答スタート']), 'center', 'center',  WhiteIndex(w));
Screen('Flip', w); %

alldataV=[];
opts.DataLines=[2 inf];
craw=size(readmatrix(filename,opts).*18.3./64,1);% Read Start row row

while 1

    if EndFlag;  break; end
    WaitSecs(0.1);
    %% store EEG(VIE ZONE)
    opts.DataLines=[craw+1 inf];
    tempdataV=readmatrix(filename,opts).*18.3./64;
    if ~isempty(tempdataV) && size(tempdataV,1)>1
        tempdataV=[tempdataV tempdataV(:,2)-tempdataV(:,1)];
        alldataV=[alldataV;tempdataV];
        craw=craw+size(tempdataV,1);

        [tempdata1,Zf1] = filter(B1f, A1f, tempdataV,Zf1); %Zf1:初期条件
        if cidx+size(tempdataV,1)-1<SAMPLE_FREQ_VIE*mtimeWindow
            cplotdata(cidx:cidx+size(tempdataV,1)-1,:)=tempdata1;
            cidx=cidx+size(tempdataV,1);
        else
            cplotdata(cidx:end,:)=tempdata1(end-(size(cplotdata,1)-cidx):end,:);
            cidx=1;
            cbaseline=nanmean(cplotdata);
        end
    end

    %% Noise Calc
    cNoiseMat=[];
    cdata=cplotdata(:,[1 2]);
    [pxx,f] = pwelch(cdata,SAMPLE_FREQ_VIE*timeWindow,0,FreqWindow,SAMPLE_FREQ_VIE);
    cNoiseMat([1 2])=sum(pxx);
    cNoiseMat([3 4])=rms(cdata);
    cNoiseMat([5 6])=max(gradient(cdata));
    %              zeroCross = cdata(1:end-1,:).*cdata(2:end,:) < 0;
    %              cNoiseMat([7 8])= sum(zeroCross)./size(cdata,1);
    cNoiseMat([9 10]) = kurtosis(cdata);
    cNoiseMatz=(cNoiseMat-NoiseMU)./NoiseSigma;
    cNoisez=[mean(abs(cNoiseMatz([1 3 5  9]))) mean(abs(cNoiseMatz([2 4 6  10])))];

    %% Display waveform
    set(titletext,'String',[ 'Click to START  EEG ' 'sec NoiseLevel L=' num2str(cNoisez(1)) '  R=' num2str(cNoisez(2))] ,'Color','w');
    set(EEGplot(1),'YData',cplotdata(:,1)'+40); hold on% freqwidthに当たる部分を表示
    set(EEGplot(2),'YData',cplotdata(:,2)'); % freqwidthに当たる部分を表示
    set(EEGplot(3),'YData',cplotdata(:,3)'-40); % freqwidthに当たる部分を表示
    figure(h);

    [x, y, buttons] = GetMouse(w);

    if any(buttons)
        break;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Rec START
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data={};TrialNo=1;
stimorder= randperm(length(imglist));

ResultPre=[];
for T=1:length(stimorder)
    opts.DataLines=[2 inf];
    craw=size(readmatrix(filename,opts).*18.3./64,1);% Read Start row
    cstim=   stimorder(T);

    A_img = imread(fullfile(stimPath, imglist{cstim}));
    A_tex = Screen('MakeTexture', w, A_img);

    alldataV=[];NoiseLog=[];NoiseFlagLog=[];
    starttime=tic;
    nextdec=0;
    SetMouse(centerX,centerY); [x,y,buttons] = GetMouse(w);
    OKFlag=0;
    while 1
        Screen('DrawTexture', w, A_tex, [], destRect);
        DrawFormattedText(w,double([num2str(T) '/' num2str(length(stimorder))  '　好みを回答してください(脳波が正常に収録されバーが赤くなったら回答可能)']), 'center',50,  WhiteIndex(w));
        %% VAS
        Screen('DrawLines', w, [centerX-Linelength/2 centerX+Linelength/2 ; rect(4)-100 rect(4)-100], LineHaba, LineColor);
        DrawFormattedText(w, double('まったく好みでない'),centerX-Linelength/2-AdjustPar-100,rect(4)-70, LineColor); % 質問左
        DrawFormattedText(w, double('とても好み'), centerX+Linelength/2+20,   rect(4)-70, LineColor); % 質問右

        [x,y,buttons] = GetMouse(w);
        if x < centerX-Linelength/2
            x= centerX-Linelength/2;
        elseif x > centerX+Linelength/2
            x=centerX+Linelength/2;
        end

        if ~OKFlag
            Screen('FillRect', w, [100 100 100], [x-2 rect(4)-120 x+2 rect(4)-80]);%% 規定時間までは回答できない
        else
            if buttons(1)
                Click=GetSecs;
                if x < centerX-Linelength/2
                    x= centerX-Linelength/2;
                elseif x > centerX+Linelength/2
                    x=centerX+Linelength/2;
                end
                Answer=x;
                PercentAns=(Answer-(centerX-Linelength/2))./(Linelength).*100;
                ResultPre(T)=PercentAns;
                Screen('Flip', w); %
                WaitSecs(0.5);
                break;
            end
            Screen('FillRect', w, [255 0 0], [x-2 rect(4)-120 x+2 rect(4)-80]);
        end
        Screen('Flip', w); %

        %% EEG STORE
        opts.DataLines=[craw+1 inf];
        tempdataV=readmatrix(filename,opts).*18.3./64;
        if ~isempty(tempdataV) && size(tempdataV,1)>1
            tempdataV=[tempdataV tempdataV(:,2)-tempdataV(:,1)];
            alldataV=[alldataV;tempdataV];
            craw=craw+size(tempdataV,1);

            [tempdata1,Zf1] = filter(B1f, A1f, tempdataV,Zf1); %Zf1:初期条件
            if cidx+size(tempdataV,1)-1<SAMPLE_FREQ_VIE*mtimeWindow
                cplotdata(cidx:cidx+size(tempdataV,1)-1,:)=tempdata1;
                cidx=cidx+size(tempdataV,1);
            else
                cplotdata(cidx:end,:)=tempdata1(end-(size(cplotdata,1)-cidx):end,:);
                cidx=1;
            end
        end

        %% Noise Calc
        cNoiseMat=[];
        cdata=cplotdata(:,[1 2]);
        [pxx,f] = pwelch(cdata,SAMPLE_FREQ_VIE*timeWindow,0,FreqWindow,SAMPLE_FREQ_VIE);
        cNoiseMat([1 2])=sum(pxx);
        cNoiseMat([3 4])=rms(cdata);
        cNoiseMat([5 6])=max(gradient(cdata));
        cNoiseMat([9 10]) = kurtosis(cdata);
        cNoiseMatz=(cNoiseMat-NoiseMU)./NoiseSigma;
        cNoisez=[mean(abs(cNoiseMatz([1 3 5  9]))) mean(abs(cNoiseMatz([2 4 6  10])))];

        %% Display waveform
        set(titletext,'String',[ ' EEG (' num2str(round(toc(starttime))) 's)    NoiseLevel L=' num2str(round(cNoisez(1),2)) '  R=' num2str(round(cNoisez(2),2))] ,'Color','w');
        set(EEGplot(1),'YData',cplotdata(:,1)'+40); hold on% freqwidthに当たる部分を表示
        set(EEGplot(2),'YData',cplotdata(:,2)'); % freqwidthに当たる部分を表示
        set(EEGplot(3),'YData',cplotdata(:,3)'-40); % freqwidthに当たる部分を表示
        drawnow;shg;

        ctime=toc(starttime);
        if length(alldataV) > SAMPLE_FREQ_VIE*timeWindow && ctime>nextdec
            %% bandpass % get power
            cdata=alldataV(end-(SAMPLE_FREQ_VIE*timeWindow)+1:end,:);
            [cdataf,Zf1]= filter(B1f,A1f, cdata,Zf1);
            [pxx,f] = pwelch(cdata,SAMPLE_FREQ_VIE*timeWindow,0,FreqWindow,SAMPLE_FREQ_VIE);

            %% Noise判定
            cNoiseMat=[];
            cNoiseMat([1 2])=sum(pxx(:,[1 2]));
            cdata=cdataf(:,[1 2]);
            cNoiseMat([3 4])=rms(cdata);
            cNoiseMat([5 6])=max(gradient(cdata));
            %              zeroCross = cdata(1:end-1,:).*cdata(2:end,:) < 0;
            %              cNoiseMat([7 8])= sum(zeroCross)./size(cdata,1);
            cNoiseMat([9 10]) = kurtosis(cdata);
            cNoiseMatz=(cNoiseMat-NoiseMU)./NoiseSigma;
            cNoisez=[mean(abs(cNoiseMatz([1 3 5  9]))) mean(abs(cNoiseMatz([2 4 6  10])))];
            if max(cNoisez)>ThuZ %%左右の電極の絶対振幅値の平均がTHを超えたら削除
                NoiseFlag=1;
            else
                NoiseFlag=0;
            end

            NoiseFlagLog=[NoiseFlagLog;NoiseFlag];
            NoiseLog=[NoiseLog;[cNoiseMatz]];

            validIdx = (NoiseFlagLog == 0) ;
            if sum(validIdx) >= 2
                OKFlag=1;
            end

            nextdec=ctime+0.5;

        end
    end
    if EndFlag;  break; end
    %         alldataV=alldataV(end-RecDuration*SAMPLE_FREQ_VIE+1:end,:);

    %% save Data
    Data{TrialNo,1} =[];
    Data{TrialNo,2} = T;
    Data{TrialNo,3} =cstim;
    Data{TrialNo,4} =imglist{cstim};
    Data{TrialNo,5} = alldataV;
    Data{TrialNo,6} = PercentAns;

    TrialNo=TrialNo+1;%次のトライアルへ
    ctime=toc(starttime);

end

DrawFormattedText(w,double(['収録終了']), 'center', 'center',  WhiteIndex(w));

save([savefilename 'Data.mat'], 'stimorder','Data','subName','selpath','imglist','IMGlist','Question','ResultPre','SAMPLE_FREQ_VIE','NoiseMetricsLabel',...
    'NoImage', 'imgpath', 'stimPath','T','savefilename','folderPath',...
    '-v7.3');

Screen('Flip', w);
WaitSecs(1)
sca
close all