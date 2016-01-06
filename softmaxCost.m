function [cost, grad] = softmaxCost(theta,numClasses, inputSize, lambda,data,labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data,2);

groundTruth = full(sparse(labels, 1:numCases, 1));


%thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%  The groundTruth matrix might come in handy.

% [nfeatures, nsamples] = size(data);
% M = theta * data;
% maxM = max(M, [], 1);
% M = bsxfun(@minus, M, maxM);
% M = exp(M);
% M = bsxfun(@rdivide, M, sum(M));
% temp = groundTruth .* log(M);
% cost = - sum(sum(temp)) ./ nsamples;
% cost = cost + sum(sum(theta .^ 2)) .* lambda ./ 2;
% 
% cost
 




M = theta*data;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
M = bsxfun(@rdivide,M,sum(M));
temp = groundTruth .* log(M);
cost = (-1/size(data,2)).* sum(sum(temp)) + (lambda/2) .* sum(sum(theta.^2));

       
       alp = groundTruth - M;
thetagrad = (-1/size(data,2)) .* (alp * data') + lambda * theta;


    







% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

