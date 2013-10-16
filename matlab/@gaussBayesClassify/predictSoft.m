function p = predictSoft(obj,Xte)
% Prob = predictSoft(obj,Xtest) : make "soft" predictions on test data with the classifier
  for i=1:length(obj.classes),             % compute probabilities for each class by Bayes rule
    p(:,i) = obj.probs(i) * evalGaussian( Xte, obj.means{i},obj.covars{i} );   % p(c) * p(x|c) 
  end;                                     
%  [tmp,c] = max(p,[],2);                   % find the index of the largest probability
%  Yte = obj.classes(c);                    % and return that class ID
end


function p = evalGaussian( X , gMean, gCov )
  d = size(X,2); n = size(X,1);               % get dimension and # of data
  p = zeros(n,1);                             % store evaluated probabilities for each datum
  constant = 1/(2*pi)^(d/2) / det(gCov)^(.5); % normalization constant for Gaussian
  invCov = inv(gCov);                         % need inverse covariance
  for i=1:size(X,1),                          % compute probability of Gaussian at each point
    p(i) = exp(-.5 * (X(i,:)-gMean)*invCov*(X(i,:)-gMean)' ) * constant; 
  end;
end
