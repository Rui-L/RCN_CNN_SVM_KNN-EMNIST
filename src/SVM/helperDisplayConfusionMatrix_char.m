function helperDisplayConfusionMatrix_char(confMat)
% Display the confusion matrix in a formatted table.

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

digits = 'A':'Z';
colHeadings = arrayfun(@(x)sprintf('%d',x),1:26,'UniformOutput',false);
format = repmat('%-9s',1,11);
header = sprintf(format,'character  |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(digits)
    fprintf('%-9s',   [digits(idx) '      |']);
    fprintf('%-9.2f', confMat(idx,:));
    fprintf('\n')
end
