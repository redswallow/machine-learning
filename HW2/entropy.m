function H=entropy(y)
% H = entropy(y)
% calculate the entropy of vector y
    yinfo=tabulate(y);
    yhist=yinfo(:,3)'/100;  %percentage of each unique value in y
    H=-yhist*log2(yhist)';