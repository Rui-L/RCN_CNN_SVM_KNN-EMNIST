% We want to pick up K images from a total of N 
% for tran_set A - Z: 
% K_A = randperm(N,K)

letter = 'Z';
Dest     = strcat('./EMNIST/selectTrain_10/',letter);
Folder = strcat('EMNIST/training/',letter);
% Dest     = './EMNIST/selectTrain_20/I';
% Folder = 'EMNIST/training/I';
FileList = dir(fullfile(Folder, '*.bmp'));
Index    = randperm(numel(FileList), 10);
for k = 1:10
  Source = fullfile(Folder, FileList(Index(k)).name)
  copyfile(Source, Dest);
end
