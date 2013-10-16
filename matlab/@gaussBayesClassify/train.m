function obj=train(obj, Xtr,Ytr, varargin)
% gbc = train(gbc, X,Y [, options...]) -- train a Bayes classifier with Gaussian class models
%   X,Y        : NxD matrix of feature values, Nx1 vector of target classes
% Options:
%  'equal'     : force all classes to share a single covariance model
%  'diagonal'  : force classes to use a diagonal covariance model
%  'weights',w : learn the model using weighted data; w should be an Nx1 vector of positive weights
%  'reg',a     : regularize by (scalar or vector) alpha added to covariance diagonal

  equalCovar = 0;                            % default to general covariances
  diagCovar = 0;
  wts = ones(size(Ytr));                     % and uniform weights
  alpha = 0;

  i=1;                                       % parse through various options
  while (i<=length(varargin)),
    switch(varargin{i})
    case 'equal', equalCovar = 1;
    case 'diagonal', diagCovar = 1;
    case 'weights', wts = varargin{i+1}; i=i+1;
    case 'reg',     alpha = varargin{i+1}; i=i+1;
    end;
    i=i+1;
  end;

  obj.classes = unique(Ytr);
  wts = wts / sum(wts);
  %nData = size(Ytr,1);
  for i=1:length(obj.classes), 
    idx = find(Ytr == obj.classes(i));
    %obj.probs(i) = length(idx)/nData;      % (the relative fraction of data in this class)
    obj.probs(i) = sum(wts(idx));           % compute the (weighted) fraction of data in class i
    %obj.means{i}  = mean( Xtr(idx,:) );    % (unweighted mean)
    obj.means{i} = wts(idx)' * Xtr(idx,:);  % compute the (weighted) mean
    tmp  = bsxfun(@minus,Xtr(idx,:),obj.means{i});  % center the data
    wtmp = bsxfun(@times,Xtr(idx,:),wts(idx));      % weighted, centered data
    if (diagCovar)
      %obj.covars{i} = diag(var( Xtr(idx,:) ));
      obj.covars{i} = diag(sum(tmp.*wtmp,1)+alpha);  % brute force weighted variance computation
    else
      %obj.covars{i} = cov( Xtr(idx,:) );
      obj.covars{i} = tmp'*wtmp + diag(alpha+0*obj.means{i});  % brute force weighted covariance computation
    end;
  end;

  if (equalCovar),                       % force covariances to be equal (take weighted average)
    Call = 0; 
    for i=1:length(obj.classes), Call = Call + obj.probs(i)*obj.covars{i}; end;
    for i=1:length(obj.classes), obj.covars{i} = Call; end;
  end;

end
