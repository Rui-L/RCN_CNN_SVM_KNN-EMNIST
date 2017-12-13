clear all
close all
%%
load emnist-letters.mat

train_img = double(dataset.train.images);
train_label = dataset.train.labels;
% train = zeros(size(train_img,1), size(train_img,2) + 1);
train = [train_label, train_img];

% train = train(1:1300,:);
% hist(train_label(1:1300),26)
% title('labels for training datas')

train = train(1:520,:); % train size, roughly equivalent to using selectTrain_20


test_img = double(dataset.test.images);
test_label = dataset.test.labels;

test = [test_label, test_img];

%%
N = 1300; % test size, roughly equivalent to using selectTest_50
K_init= randperm(length(test_label),N);
label_for_test = test(K_init,1);
hist(label_for_test,26)
title('label for test datas')

K = zeros(N,1);
% d_knn_l2 = cell(N,1);
d_knn_l1 = cell(N,1);

% for m = 1:N
%     d_knn_l2{m} = zeros(size(train,1),1);
%     K(m) = K_init(1,m);
% %     for l = 1:size(train,1)
%         for j = 1:size(train,2)-1
%             d_knn_l2{m}(:,1) = d_knn_l2{m}(:,1) + sqrt(train(:,j+1).^2 - ones(size(train,1),1)*test(K(m),j+1).^2);
%         end
%     %end
% end


for m = 1:N
    d_knn_l1{m} = zeros(size(train,1),1);
    K(m) = K_init(1,m);
%     for l = 1:size(train,1)
        for j = 1:size(train,2)-1
            d_knn_l1{m}(:,1) = d_knn_l1{m}(:,1) + abs(train(:,j+1) - ones(size(train,1),1)*test(K(m),j+1));
        end
    %end
end

%% case 1: K = 1
% for m = 1:N  % use L2 distance
%     % find the min value of d_knn for each test data and its index 
%     [val_min(m,1),ind_min(m,1)] = min(d_knn_l2{m});
%     % find the 1-nearist neighbor for each test in train, and extract the
%     % label of it
%     label_k1_l2(m,1) = train(ind_min(m,1),1);
% end
% 
% % find error rate:
% error_k1_l2 = 0;
% for m = 1:N
%     if label_k1_l2(m) ~= test(K(m),1)
%         error_k1_l2 = error_k1_l2+1;
%     end
% end
% accuracy_rate_k1_l2 = 1 - error_k1_l2/N

for m = 1:N % use L1 distance
    % find the min value of d_knn for each test data and its index 
    [val_min(m,1),ind_min(m,1)] = min(d_knn_l1{m});
    % find the 1-nearist neighbor for each test in train, and extract the
    % label of it
    label_k1_l1(m,1) = train(ind_min(m,1),1);
end

% find error rate:
error_k1_l1 = 0;
for m = 1:N
    if label_k1_l1(m) ~= test(K(m),1)
        error_k1_l1 = error_k1_l1+1;
    end
end
accuracy_rate_k1_l1 = 1 - error_k1_l1/N  
%% case 2: K = 5
% temp for d_knn
clear ind_min
d_knn_temp = d_knn_l1;
for kc = 1:5
    for m = 1:N    
        % find the 5 nearist neighbors of each test
        [val_min(m,kc),ind_min(m,kc)] = min(d_knn_temp{m});
        label_k5_temp_l1(m,kc) = train(ind_min(m,kc),1);
        d_knn_temp{m}(ind_min(m,kc),1) = 10000000;
    end
end

% find the label based on most nearist neighbors
count = zeros(N,1);
label_ind = zeros(N,1);
for m = 1:N
    for kc = 2:5
        if label_k5_temp_l1(m,kc) == label_k5_temp_l1(m,1)
            count(m) = count(m) + 1;
        else
            tp(m,1) = 1;
            label_ind(m,tp(m)) = label_k5_temp_l1(m,kc);
            tp(m,1) = tp(m,1) +1;
        end
    end
end
% determine whether the majority of label_k5_temp(m,:) is equal to 
% label_k5_temp(m,1):
for m = 1:N
    if count(m) > 2 
        label_k5_l1(m,1) = label_k5_temp_l1(m,1);
    else
        label_k5_l1(m,1) = label_ind(m);
    end
end
        
% find error rate:
error_k5_l1 = 0;
for m = 1:N
    if label_k5_l1(m) ~= test(K(m),1)
        error_k5_l1 = error_k5_l1+1;
    end
end
accuracy_rate_k5_l1 = 1 - error_k5_l1/N  

%% for case 3 (K = 9):

nK = 9;

% for K = 9;
d_knn_temp = d_knn_l1;
for kc = 1:nK
    for m = 1:N    
        % find the 5 nearist neighbors of each test
        [val_min(m,kc),ind_min(m,kc)] = min(d_knn_temp{m});
        label_k9_temp_l1(m,kc) = train(ind_min(m,kc),1);
        d_knn_temp{m}(ind_min(m,kc),1) = 1000000;
    end
end

% find the label based on most nearist neighbors
count = zeros(N,1);
label_ind = zeros(N,1);
for m = 1:N
    for kc = 2:nK
        if label_k9_temp_l1(m,kc) == label_k9_temp_l1(m,1)
            count(m) = count(m) + 1;
        else
            tp(m,1) = 1;
            label_ind(m,tp(m)) = label_k9_temp_l1(m,kc);
            tp(m,1) = tp(m,1) +1;
        end
    end
end
% determine whether the majority of label_k5_temp(m,:) is equal to 
% label_k5_temp(m,1):
for m = 1:N
    if count(m) > nK
        label_k9_l1(m,1) = label_k9_temp_l1(m,1);
    else
        if tp(m) == 2
            label_k9_l1(m,1) = label_ind(m,1);
        else
            for p = 1:tp(m)-1
                if length(find(label_ind(m,p))) > nK(1)/2
                    label_k9_l1(m,1) = label_ind(m,p);
                end
            end
        end
    end
end

% find error rate:
error_k9_l1 = 0;
for m = 1:N
    if label_k9_l1(m) ~= test(K(m),1)
        error_k9_l1 = error_k9_l1+1;
    end
end
accuracy_rate_k9_l1 = 1 - error_k9_l1/N
