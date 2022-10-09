% MATLAB ver. 2021b
% using deep learning toolbox & parallel computing toolbox

load('IB-TAS_A') % or load('IB-TAS_B')
 
% IB-TAS_A includes following:

% NF: number of features
% NA: number of antennas at Alice
% NB: number of antennas at Bob
% NE: number of antennas at Eve
% pB: SNR at Bob
% pE: SNR at Eve
% M: number of classes
% XTrain_A (features x samples): normalized feature vectors used for training by ML-TAS
% XTest_A (features x samples): normalized feature vectors used for testing by ML-TAS
% YTrain_A (label x samples): class label to which feature vectors are mapped based on the optimal transmit antenna determined in IB-TAS
% YTest_A (label x samples): class label to which feature vectors are mapped based on the optimal transmit antenna determined in IB-TAS

% IB-TAS_B includes following:

% NF: number of features
% NA: number of antennas at Alice
% NB: number of antennas at Bob
% NE: number of antennas at Eve
% pB: SNR at Bob
% pE: SNR at Eve
% M: number of classes
% XTrain_B (features x samples): normalized feature vectors used for training by ML-TAS
% XTest_B (features x samples): normalized feature vectors used for testing by ML-TAS
% YTrain_B (label x samples): class label to which feature vectors are mapped based on the optimal transmit antenna determined in IB-TAS
% YTest_B (label x samples): class label to which feature vectors are mapped based on the optimal transmit antenna determined in IB-TAS



%% NN-based TAS
% 'modelGradients.m' should be used. 
Inputs_XTrain = reshape(XTrain,1,1, size(XTrain, 2),size(XTrain, 1));
Inputs_Test1 = reshape(XTest,1,1, size(XTest, 2),size(XTest, 1));
classes = string(1:1:size(M,1));

units = floor(NF*2/3);
layers = [
    imageInputLayer([1 1 size(Inputs_XTrain,3)],"Name","Input data","Normalization","none");
    fullyConnectedLayer(units, "Name","fc1")
    reluLayer("Name","relu1")
    fullyConnectedLayer(NA, "Name","fc2")];
NNet=dlnetwork(layerGraph(layers));

numEpochs=100;
miniBatchSize = 256;
numIterationsPerEpoch = floor(N_samples/miniBatchSize);
executionEnvironment = "auto";
averageGrad=[];
averageSqGrad=[];
iteration=1;
learnRate = 0.0001;
gradDecay = 0.9;
sqGradDecay = 0.95;

plots = "training-progress";
if plots == "training-progress"
    figure
    lineLossTrain = animatedline;
    xlabel("Total Iterations")
    ylabel("Loss")
end

% Train NN
for epoch = 1:numEpochs
    
    idx = randperm(size(YTrain,2));
    Inputs_XTrain = Inputs_XTrain(:,:,:,idx);
    YTrain = YTrain(:,idx);
    
    for i = 1:numIterationsPerEpoch
        
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X = Inputs_XTrain(:,:,:,idx);
        Y = YTrain(:,idx);
                
        dlX = dlarray(single(X),'SSCB');
        
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        [grad,loss] = dlfeval(@modelGradients,NN_Net,dlX,Y);
        [NN_Net,averageGrad,averageSqGrad] = adamupdate(NN_Net,grad,averageGrad,averageSqGrad,iteration);
        
        if plots == "training-progress"
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Loss During Training: Epoch - " + epoch + "; Iteration - " + i)
            drawnow
        end
        
        iteration = iteration + 1;
    end
end

% Test NN
dlXTest = dlarray(XTest,'SSCB');
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlXTest = gpuArray(dlXTest);
end

dlYPred = predict(NN_Net,dlXTest);
[~,Pred_NN] = max(extractdata(dlYPred),[],1);

%% SVM-based TAS
% Train SVM
temp=templateSVM('Standardize',false,'KernelFunction','rbf','KernelScale','auto');
options = statset('UseParallel',true,'MaxIter',100);
SVMmodel = fitcecoc(XTrain,YTrain,'Learners',temp,'Coding','onevsall','Options',options,'OptimizeHyperparameters','auto');

% Test SVM
Pred_SVM=predict(SVMmodel,XTest);

%% NB-based TAS
% Train NB
NBmodel=fitcnb(XTrain,YTrain,'DistributionNames','normal');

% Test NB
Pred_NB=predict(NBmodel,XTest);

