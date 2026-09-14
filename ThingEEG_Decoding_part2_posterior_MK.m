%% ThingEEG Binary Decoding: Animal vs. Non-Animal
% Part 2: posterior analysis

clear all
close all


%% Load data
load('S12_epoched.mat');           % → EEG_epoch
load('S12_epoched_eventlist.mat'); % → neweventlist
T = readtable('object_concepts.xlsx');       % concept info

rng(40);
clearvars -except EEG_epoch neweventlist T DA_sh
times = EEG_epoch.times;

%% Get trials of interest
% Our code assumes that the eeg data are formatted as trials x electrodes x
% time
% So we reformat these data to match that order
data2 = permute(EEG_epoch.data, [3 1 2]);   % [trials × elec × time]

%% Part 1: Define categories (animal vs. non-animal)
% Animal class
animalLocs  = find(strcmp(T.Bottom_upCategory_HumanRaters_, 'animal'));
insectLocs  = find(strcmp(T.Bottom_upCategory_HumanRaters_, 'insect'));
birdLocs    = find(strcmp(T.Bottom_upCategory_HumanRaters_, 'bird'));

% Non-animal class
toolLocs    = find(strcmp(T.Bottom_upCategory_HumanRaters_, 'tool'));
fruitLocs   = find(strcmp(T.Bottom_upCategory_HumanRaters_, 'fruit'));
vegetLocs   = find(strcmp(T.Bottom_upCategory_HumanRaters_, 'vegetable'));
vehicleLocs = find(strcmp(T.Bottom_upCategory_HumanRaters_, 'vehicle'));

% For ease of manipulation we store these location into a structure called
% allCateg
% Part 1: merge sub-categories into the two target classes
allCateg = struct();

% Part 1: add animal and non-animal categories to the structure
allCateg.animal    = unique([animalLocs; insectLocs; birdLocs]);
allCateg.nonanimal = unique([toolLocs; fruitLocs; vegetLocs; vehicleLocs]);

%% Find trial indices for each category
[labelsOut, indices, ~] = findPresentationIndices_LS(allCateg, T, neweventlist);

categoryLabels = {'animal', 'nonanimal'};
nClasses = 2;

% keep only relevant trials
data_sel = data2(indices,:,:); 

dataCell = cell(nClasses, 2);
[dataCell{:,2}] = deal('animal', 'nonanimal');
for c = 1:nClasses
    flagTrials = ismember(labelsOut(:,2), c);
    dataCell{c,1} = single(data_sel(flagTrials, :, :));
end

fprintf('Class sizes  –  animal: %d   non-animal: %d\n', ...
    size(dataCell{1,1},1), size(dataCell{2,1},1));

%% Now we need to make sure that we have the same amount of trials in our categories

clear newdataCell
coi = 1:2;
minC = min(histc(labelsOut(:,2),coi));

% Let's reduce the number of trials just for practice
% Part 1: commented this out to get more accurate model
%if minC > 300
 %   minC = 100;
%end

% newdataCell: animalClass and nonanimalClass
[newdataCell{1:2,2}]=deal('animalClass','nonanimalClass');

% Equalize number of trials across categories of interest
for c = 1:length(coi)
    origdata = dataCell{coi(c),1};
    subdata = origdata(randperm(size(origdata,1),minC),:,:);
    newdataCell{c,1} = subdata;
end

%% Part 2: restrict the analysis to POSTERIOR electrodes only
% label used in the figure title to flag the ROI
roiLabel = 'posterior';

% get the channel names from the EEG structure
chanLabels = {EEG_epoch.chanlocs.labels};

% posterior ROI (electrodes taken from the figure)
roiNames = {'PO7','PO3','POz','PO4','PO8','O1','Oz','O2'};

% find indices of the ROI electrodes
roiIdx = find(ismember(chanLabels, roiNames));


fprintf('Posterior ROI: %d electrodes used (%s)\n', numel(roiIdx), strjoin(roiNames,', '));

% keep only the ROI electrodes (this propagates through the whole pipeline)
for c = 1:nClasses
    newdataCell{c,1} = newdataCell{c,1}(:, roiIdx, :);
end

%% We specify a few parameters

nIterations = 10; % Part 1
testRatio =  0.2; % Part 1: commonly used 80-20 split 
conditionM = 2; % Part 1: two classes to decode
n_folds = 5; % Part 1: k-fold cross validation

%  Split minC into training (1-testRatio) and test (testRatio).
%  The test pool is split into non-overlapping pseudo-trials of size Npseudo.
nTest       = round(minC * testRatio);
nTrain      = minC - nTest;
Npseudo     = 3;  % Part 1: trials averaged per pseudo-trial
nPseudo_tst = floor(nTest / Npseudo);   % Part 1: pseudo-trials per class in test set

fprintf('Train trials: %d  |  Test pool: %d  →  %d pseudo-test trials/class\n', ...
    nTrain, nTest, nPseudo_tst);

% To make it faster we can downsample and only test every other point

tStep       = 2;       % 1 = no downsampling, 2 = every other sample
timeSamps = (1:tStep:length(times))';
nTS       = length(timeSamps);
times_ds  = times(timeSamps);
nIterations = 10

% We will run the decoding 10 times, each iteration will test all time points
DA = nan(nIterations, nTS); % Part 1: DA: repetitions x time


for iterX = 1:nIterations

    clear pseudoData
    % Shuffle trials independently per classnIterations
    pseudoData = nan(nClasses, minC, ...
                     size(newdataCell{1,1}, 2), ...
                     size(newdataCell{1,1}, 3), 'single');
    for c = 1:nClasses
        cm = newdataCell{c,1};
        pseudoData(c,:,:,:) = cm(randperm(minC), :, :);
    end

    % 5-fold cross-validation for this iteration
        cvA = cvpartition(minC, 'KFold', n_folds);
    cvB = cvpartition(minC, 'KFold', n_folds);

    % accuracy per fold, averaged into DA at the end of the iteration
    acc_fold = nan(n_folds, nTS);
    for fold = 1:n_folds

    idx_train_A = training(cvA, fold);
    idx_test_A  = test(cvA, fold);
    idx_train_B = training(cvB, fold);
    idx_test_B  = test(cvB, fold);

    for t = 1:nTS
        ts = timeSamps(t);

        %% Training data (class A = animal, class B = non-animal)
        trPoolA = double(squeeze(pseudoData(1, idx_train_A, :, ts)));
        trPoolB = double(squeeze(pseudoData(2, idx_train_B, :, ts)));
        nPsTr   = min(floor(size(trPoolA,1)/Npseudo), floor(size(trPoolB,1)/Npseudo));
        trainA  = nan(nPsTr, size(trPoolA,2));
        trainB  = nan(nPsTr, size(trPoolB,2));
        for ps = 1:nPsTr
            idx_ps       = (ps-1)*Npseudo + (1:Npseudo);
            trainA(ps,:) = mean(trPoolA(idx_ps, :), 1);
            trainB(ps,:) = mean(trPoolB(idx_ps, :), 1);
        end
        trainX = [trainA; trainB];
        trainY = [ones(size(trainA,1),1); 2*ones(size(trainB,1),1)];

        % Part 1: Normalization: min-max computed on training data only
        minTr  = min(trainX);
        maxTr  = max(trainX);
        rng_tr = maxTr - minTr;
        rng_tr(rng_tr == 0) = 1;          % avoid division by zero
        trainX_norm = (trainX - minTr) ./ rng_tr;

        %% Part 1: Pseudo-averaging: average Npseudo consecutive test
        % trials to raise SNR
        poolA  = double(squeeze(pseudoData(1, idx_test_A, :, ts)));
        poolB  = double(squeeze(pseudoData(2, idx_test_B, :, ts)));

        nPsA   = floor(size(poolA,1) / Npseudo);
        nPsB   = floor(size(poolB,1) / Npseudo);
        nPs    = min(nPsA, nPsB);   % use same count for balance

        testA  = nan(nPs, size(poolA,2));
        testB  = nan(nPs, size(poolB,2));
        for ps = 1:nPs
            idx_ps        = (ps-1)*Npseudo + (1:Npseudo);
            testA(ps,:)   = mean(poolA(idx_ps, :), 1);
            testB(ps,:)   = mean(poolB(idx_ps, :), 1);
        end

        % Apply same normalization parameters from training
        testX      = [testA; testB];
        testX_norm = (testX - minTr) ./ rng_tr;
        testY      = [ones(nPs,1); 2*ones(nPs,1)];

        %% Part 1: Train & test SVM 
        mdl    = fitcsvm(trainX_norm, trainY, ...
                         'KernelFunction', 'linear', ...
                         'BoxConstraint', 1, ...
                         'Standardize', false); % normalize manually
        predY  = predict(mdl, testX_norm);
        acc_fold(fold, t) = mean(predY == testY) * 100; % store accuracy
    end
    end                                   % end fold loop
    DA(iterX, :) = nanmean(acc_fold, 1);  % average across the 5 folds

    fprintf('  Iteration %d/%d done\n', iterX, nIterations);
end

% Average across iterations
accuracy_real = nanmean(DA, 1);   % [1 × nTS]


%% Part 1: Permutation test (shuffle labels to build null distribution)
nPerms = 10;
DA_sh = nan(nPerms, nTS);

fprintf('\n--- Permutation test (%d permutations) ---\n', nPerms);

for p = 1:nPerms

    % Combine all trials from both classes, then randomly reassign labels
    allTrials = cat(1, newdataCell{1,1}, newdataCell{2,1});   % [2*minC × elec × time]
    nTotal    = size(allTrials, 1);
    permIdx   = randperm(nTotal);
    allTrials_perm = allTrials(permIdx, :, :);

    % Re-split into two "fake" classes of equal size
    permData = cell(nClasses, 1);
    permData{1} = allTrials_perm(1:minC, :, :);
    permData{2} = allTrials_perm(minC+1:2*minC, :, :);

    % 5-fold cross-validation (replaces hold-out); accuracy averaged over folds -> one value per permutation
    n_folds = 5;
    cvP1 = cvpartition(minC, 'KFold', n_folds);
    cvP2 = cvpartition(minC, 'KFold', n_folds);
    acc_inner = nan(n_folds, nTS);
    for fold = 1:n_folds

        idx_tr1 = training(cvP1, fold); idx_te1 = test(cvP1, fold);
        idx_tr2 = training(cvP2, fold); idx_te2 = test(cvP2, fold);

    for t = 1:nTS
        ts = timeSamps(t);

        trPoolA = double(squeeze(permData{1}(idx_tr1, :, ts)));
        trPoolB = double(squeeze(permData{2}(idx_tr2, :, ts)));
        nPsTr   = min(floor(size(trPoolA,1)/Npseudo), floor(size(trPoolB,1)/Npseudo));
        trainA  = nan(nPsTr, size(trPoolA,2));
        trainB  = nan(nPsTr, size(trPoolB,2));
        for ps = 1:nPsTr
            idx_ps       = (ps-1)*Npseudo + (1:Npseudo);
            trainA(ps,:) = mean(trPoolA(idx_ps, :), 1);
            trainB(ps,:) = mean(trPoolB(idx_ps, :), 1);
        end
        trainX = [trainA; trainB];
        trainY = [ones(size(trainA,1),1); 2*ones(size(trainB,1),1)];

        % Normalize
        minTr  = min(trainX);  maxTr = max(trainX);
        rng_tr = maxTr - minTr; rng_tr(rng_tr==0) = 1;
        trainX_norm = (trainX - minTr) ./ rng_tr;

        % Pseudo-averaged test
        poolA = double(squeeze(permData{1}(idx_te1, :, ts)));
        poolB = double(squeeze(permData{2}(idx_te2, :, ts)));
        nPs   = min(floor(size(poolA,1)/Npseudo), floor(size(poolB,1)/Npseudo));

        testA = nan(nPs, size(poolA,2));
        testB = nan(nPs, size(poolB,2));
        for ps = 1:nPs
            idx_ps      = (ps-1)*Npseudo + (1:Npseudo);
            testA(ps,:) = mean(poolA(idx_ps,:),1);
            testB(ps,:) = mean(poolB(idx_ps,:),1);
        end

        testX      = [testA; testB];
        testX_norm = (testX - minTr) ./ rng_tr;
        testY      = [ones(nPs,1); 2*ones(nPs,1)];

        mdl   = fitcsvm(trainX_norm, trainY, ...
                        'KernelFunction','linear','BoxConstraint',1,'Standardize',false);
        predY = predict(mdl, testX_norm);
        acc_inner(fold, t) = mean(predY == testY) * 100;
        end                          

    end 
    DA_sh(p, :) = nanmean(acc_inner, 1); 

    fprintf('  Permutation %d/%d done\n', p, nPerms);
end

% Null distribution summary
perm_mean   = nanmean(DA_sh, 1);
permCI_low  = prctile(DA_sh,  2.5, 1);   % Part 1: lower boundary of 95%-CI
permCI_high = prctile(DA_sh, 97.5, 1);   % Part 1: upper boundary of 95%-CI

%% Significance  (real accuracy above upper 95% CI bound of null)
sig_mask = accuracy_real > permCI_high;    % logical [1 × nTS]

%% Plotting
% Part 1: i replaced the previous plot sections with this one

figure('Name', 'EEG Decoding: Animal vs. Non-Animal', ...
       'Position', [100 100 900 500]);

% Shaded permutation distribution
tRow = times_ds(:)'; % transpose vector
fill([tRow, fliplr(tRow)], ...
     [permCI_low, fliplr(permCI_high)], ...
     [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on; 

% Chance level
yline(50, 'k--');

% Permutation mean
plot(tRow, perm_mean, 'k-', 'LineWidth', 1, ...
     'DisplayName', 'Perm. mean');

% Real accuracy
plot(tRow, accuracy_real, 'r-', 'LineWidth', 2, ...
     'DisplayName', 'Real accuracy');

% Mark significant time points
if any(sig_mask)
    scatter(tRow(sig_mask), ...
            accuracy_real(sig_mask), ...
            20, [0.55 0 0], 'filled', 'DisplayName', 'p < 0.05 (perm.)');
end

xlim([-100 600]);
ylim([30 100]);
xlabel('Time (ms)');
ylabel('Decoding accuracy (%)');
subjectID = 'S12';
% Part 2: title shows the posterior ROI instead of "global"
title(sprintf('Animal vs. Non-Animal  |  ROI: %s  |  Pseudo-averaged', roiLabel));
legend('95% CI', 'Chance', 'Null mean', 'Real accuracy', ...
       'Significant', 'Location', 'northwest');
grid on;
hold off;

fprintf('\nDone. Peak accuracy: %.1f%% at %.0f ms\n', ...
        max(accuracy_real), tRow(accuracy_real == max(accuracy_real)));

%% save
% Part 2: separate output filenames so Part 1 results are not overwritten
save('DA_part2_posterior.mat','DA');
save('DA_sh_part2_posterior.mat','DA_sh');
