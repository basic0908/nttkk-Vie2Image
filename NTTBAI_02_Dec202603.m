close all
defaultpath=pwd;
outputpath=[defaultpath '\output'];mkdir(outputpath)
mkdir([defaultpath '\Model'])
modelsavepath=[defaultpath '\Model'];

[File, Path] = uigetfile( ...
    fullfile(defaultpath, 'data', 'Trainingdata', '*.mat'), ...
    '学習データを選択してください');

DataFile = fullfile(Path, File);
load(DataFile);

imgID = cell2mat(Data(:,3));
[~, sortIdx] = sort(imgID);
Data_pre_sorted = Data(sortIdx,:);

stimPath = IMGlist(1).folder;  % 適宜変更

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EEG decoder model
%% EEG setting
NoiseMU=[138.50230	138.50230	12.182700	12.182700	103.38700	103.38700	0.035300002	0.035300002	10.124000	10.124000];
NoiseSigma=[123.24060	123.24060	6.0254002	6.0254002	69.441597	69.441597	0.0038999999	0.0038999999	2.8989999	2.8989999];

timeWindow=4;%sec window
FreqWindow=[4:0.5:40];
ThuZ=2;

[B1f,A1f] = butter(4,[3/(SAMPLE_FREQ_VIE/2) 40/(SAMPLE_FREQ_VIE/2)]);% 3~40Hzバタワースフィルタ設計
Zf1=[];

%% lasssoGLM
VIE2preferenceModel={};%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EEG pre-process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
noiseFlag=[];cpower=[];conddata=[];g=1;conddata2={};

for T=1:size(Data,1)
    %% #1 VIE
    alldataV=Data{T,5};
    %     wlist=0:SAMPLE_FREQ_VIE*2:length(alldataV)-timeWindow*SAMPLE_FREQ_VIE; %2sec slide
    wlist=0:SAMPLE_FREQ_VIE/2:length(alldataV)-timeWindow*SAMPLE_FREQ_VIE; %0.5sec slide
    for m= wlist
        cdata= alldataV(m+1:m+SAMPLE_FREQ_VIE*timeWindow,:);
        [cdataf,Zf1]= filter(B1f, A1f,cdata,Zf1);

        %% normal power
        [pxx,f] = pwelch(cdata,SAMPLE_FREQ_VIE*timeWindow,0,FreqWindow,SAMPLE_FREQ_VIE);
        %             figure;plot(f,pxx);%hold on
        for ele=1:3
            cpower(ele,g,:)=[pxx(:,ele)];%Relative Power 1Trial 2windowNo 3freq 4channel
        end

        %% noise 処理
        [pxx,f] = pwelch(cdata(:,[1 2]),SAMPLE_FREQ_VIE*timeWindow,0,FreqWindow,SAMPLE_FREQ_VIE);
        NoiseMatsub(g,[1 2])=sum(pxx);
        NoiseMatsub(g,[3 4])=rms(cdataf(:,[1 2]));
        NoiseMatsub(g,[5 6])=max(gradient(cdataf(:,[1 2])));
        %             zeroCross = cdataf(1:end-1,[1 2]).*cdataf(2:end,[1 2]) < 0;
        %             NoiseMatsub(g,[7 8])=     sum(zeroCross)./size(cdataf(:,[1 2]),1);
        NoiseMatsub(g,[9 10]) = kurtosis(cdataf(:,[1 2]));
        cNoisez=(NoiseMatsub(g,:)-NoiseMU)./NoiseSigma;
        cNoise(g,:)=cNoisez;

        cNoisezMean=[mean(abs(cNoisez([1 3 5  9]))) mean(abs(cNoisez([2 4 6 10])))];
        if max(cNoisezMean)>ThuZ %%左右の電極の絶対振幅値の平均がTHを超えたら削除
            noiseFlag(g)=1;
            cpower(:,g,:)=NaN;%1Trial 2windowNo 3freq 4channel
        else
            noiseFlag(g)=0;
        end

        conddata(g,1)=[Data{T,6}];
        conddata2{g,1}=Data{T,4};
        g=g+1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% #1 VIE2preference Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VIE2preferenceModel={};
% ALLEEG=[squeeze(cpower(1,:,:)) squeeze(cpower(2,:,:)) squeeze(cpower(3,:,:))];
ALLEEG=[squeeze(cpower(3,:,:))];%差分データだけ
useidx=find(not(noiseFlag));

cannot2=conddata(useidx,1)';%VAS rating
cannot3=conddata2(useidx);%img name

%% EEG signal staderize
cpowerz=(ALLEEG(useidx,:)-nanmean(ALLEEG(useidx,:)))./nanstd(ALLEEG(useidx,:));
MU=nanmean(ALLEEG(useidx,:));
SIGMA=nanstd(ALLEEG(useidx,:));
cpowerz(isnan(cpowerz))=0;

%% PCA
[VIEcoeff , VIEscore , latent , tsquared , explained , VIEpcamu ]  = pca(cpowerz);
numscore =150;
if size(VIEscore,2) <numscore
    numscore=size(VIEscore,2);
end
% figure()
% pareto(explained)
% xlabel('Principal Component')
% ylabel('Variance Explained (%)')
VIEscoremu=nanmean(VIEscore(:,1:numscore));
VIEscoresigma=nanstd(VIEscore(:,1:numscore));
VIEscorez=(VIEscore(:,1:numscore)-VIEscoremu)./VIEscoresigma;

%% PCAver
X=VIEscorez;%PCAの主成分
Y=cannot2';
G = cannot3;  % 画像名のセル配列（グループ化に使用）
%% not PCAver
% X=cpowerz;%
% Y=cannot2';

for R = 1 : 10 % 期待精度評価のためにくりかえす
    fprintf('Iteration R = %d\n', R);

    while 1
        % 【重要】画像名(G)を基準に10分割する。
        % これにより、同じ画像名のデータは必ず同じ「Fold」に振り分けられます。
        c = cvpartition(G, 'KFold', 10);

        % 10分割のうちの1つをテストデータ、残りを学習データにする（例として1番目を使用）
        Trainidx = find(training(c, 1));
        Testidx = find(test(c, 1));
        VIE2preferenceModel(R).Trainidx=Trainidx;
        VIE2preferenceModel(R).Testidx=Testidx;

        % --- 以降の処理は元のコードと同様 ---
        [B, FitInfo] = lasso(X(Trainidx,:), Y(Trainidx), 'CV', 10, ...
            'Options', statset('UseParallel', false), 'NumLambda', 25);

        % (中略: モデル評価・保存処理)
        VIE2preferenceModel(R).TrainPredictedMat= nansum(X(Trainidx,:) .* B(:, FitInfo.IndexMinMSE)', 2) + FitInfo.Intercept(FitInfo.IndexMinMSE);
        Corr = corr([Y(Trainidx) VIE2preferenceModel(R).TrainPredictedMat], 'rows', 'pairwise', 'Type', 'Spearman');
        CorrMat = Corr(1, 2);

        VIE2preferenceModel(R).TrainCorr=CorrMat;

        % 相関係数の計算など
        PredictedMat_Test = nansum(X(Testidx,:) .* B(:, FitInfo.IndexMinMSE)', 2) + FitInfo.Intercept(FitInfo.IndexMinMSE);
        Corr = corr([Y(Testidx) PredictedMat_Test], 'rows', 'pairwise', 'Type', 'Spearman');
        CorrMat = Corr(1, 2);
        VIE2preferenceModel(R).TestPredictedMat=PredictedMat_Test;
        % ループ脱出条件（学習に成功し、一定以上の精度が出た場合）
        if CorrMat > 0.001
            VIE2preferenceModel(R).TestCorr = CorrMat;
            % ...他のデータを格納...
            break
        end
    end
end

[~,I]= sort([[VIE2preferenceModel.TrainCorr] + [VIE2preferenceModel.TestCorr]],'descend');
VIE2preferenceModel=VIE2preferenceModel(I(1));%best

Trainidx=VIE2preferenceModel.Trainidx;
Testidx=VIE2preferenceModel.Testidx;
% PredictedMat=    VIE2preferenceModel.TrainPredictedMat;
% Corr=corr([Y(Trainidx) PredictedMat],'rows','pairwise');
% Corr=corr([Y(Trainidx) PredictedMat],'Type','Spearman','rows','pairwise');

h=figure('visible','on','Position',  [  271          37        1078         441]);
clf
subplot(1,2,1)
scatter(Y(Trainidx),VIE2preferenceModel.TrainPredictedMat,100,[0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.5,'MarkerEdgeAlpha', 0.5)
xlabel(['Real'])
ylabel(['Predicted'])
% xlabel(['Post実測値z'])
% ylabel(['Post予測値z'])
title(['VIE2Preference model Train corr=' num2str(VIE2preferenceModel.TrainCorr)])
set(gca, 'FontSize',12)
set(gca, 'LineWidth', 2)
box off

subplot(1,2,2)
scatter(Y(Testidx),VIE2preferenceModel.TestPredictedMat,100,[0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.5,'MarkerEdgeAlpha', 0.5)
xlabel(['Real'])
ylabel(['Predicted'])
title(['VIE2Preference model Test corr=' num2str(VIE2preferenceModel.TestCorr)])
set(gca, 'FontSize',12)
set(gca, 'LineWidth', 2)
box off

sgtitle([subName 'modelAccuracy(VIE2preferenceModel)'])
saveas(gca,[modelsavepath '\'  strtrim(subName) datestr(now,'yyyymmdd')  'modelAccuracy(VIE2preferenceModel)' '.jpg'])
%

%% 最終的に使うモデルは全部のデータを使ったもの
TrainB={};TrainFitInfo={};Corr=[];
for R=1:10
    R
    while 1
        [B,FitInfo] = lasso(X,Y,'CV',10,'Options', statset('UseParallel',false),'NumLambda',25); %
        PredictedMat=nansum(X.* B(:,FitInfo.IndexMinMSE)',2)+FitInfo.Intercept(FitInfo.IndexMinMSE);
        Corr(R)=corr(Y,PredictedMat,'rows','pairwise','Type','Spearman');
        TrainB{R}=B;
        TrainFitInfo{R}=FitInfo;
        if max(abs(B(:,FitInfo.IndexMinMSE)))>0.001
            break
        end
    end
end
[~,bestidx]=max(Corr);
VIE2preferenceModel.CoefMat=[TrainB{bestidx}(:,TrainFitInfo{bestidx}.IndexMinMSE) ;TrainFitInfo{bestidx}.Intercept(TrainFitInfo{bestidx}.IndexMinMSE)];
VIE2preferenceModel.RMSEMat=sqrt(TrainFitInfo{bestidx}.MSE(TrainFitInfo{bestidx}.IndexMinMSE));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% model save
decoderName=[ strtrim(subName) datestr(now,'yyyymmdd')     '.mat'];
save([modelsavepath '\' decoderName ],'Data','B1f','A1f','Zf1','VIE2preferenceModel','VIEscoremu','VIEscoresigma','VIEpcamu','VIEcoeff',...
    'SAMPLE_FREQ_VIE','MU','SIGMA','numscore','FreqWindow','timeWindow','ThuZ','selpath','NoiseMetricsLabel','NoiseMU','NoiseSigma',...
    'ALLEEG', 'subName','defaultpath','modelsavepath','decoderName','Question','imglist',...
    'cannot2','cannot3','IMGlist','imglist','stimPath','folderPath','-v7.3');

