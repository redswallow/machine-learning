function  IG=ig(y,x)
    for i=unique(x)'
        xP=length(x(find(x==i))')/length(x)
        xEntropy=entropy(y(find(x==i)));
        H=xP*xEntropy;
    end;
    IG=entropy(y)-sum(H);