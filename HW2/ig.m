function IG=ig(y,x)
% IG = ig(y,x)
% calculate the information gian: IG(y)=H(y)-H(y,x); y,x are two vectors
    xP=[];xEntropy=[];
    for i=unique(x)'
        xP=[xP length(x(x==i)')/length(x)];
        xEntropy=[xEntropy entropy(y(x==i))];
    end;
    IG=entropy(y)-xP*xEntropy';