function H=entropy(y)
        yinfo=tabulate(y);
        yhist=yinfo(:,3)'/100;
        H=-yhist*log2(yhist)';