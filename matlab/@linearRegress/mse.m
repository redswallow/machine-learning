function e = mse(obj,Xte,Yte)             
% err = mse(obj, X,Y) : compute the mean squared error of predictor "obj" on test data (X,Y)
  e = mean( (Yte - predict(obj,Xte)).^2 );  
end
