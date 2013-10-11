function IG=ig(y,x)
% IG = ig(y,x)
% calculate the information gian: IG(y)=H(y)-H(y,x); y,x are two vectors
    H=0;
    for i=unique(x)'
        xP=length(x(find(x==i))')/length(x);
        xEntropy=entropy(y(find(x==i)));
        H=H+xP*xEntropy;
    end;
    IG=entropy(y)-H;