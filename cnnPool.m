function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);
pooledFeatures = zeros(numFeatures, numImages, floor(convolvedDim / poolDim), floor(convolvedDim / poolDim));

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------
temp=zeros(floor(convolvedDim / poolDim),floor(convolvedDim / poolDim));
for i=1:numImages
    for j=1:numFeatures
        cf=squeeze(convolvedFeatures(j,i,:,:));
        for k=1:floor(convolvedDim / poolDim)
             for l=1:floor(convolvedDim / poolDim)
             offset1=(k-1)*poolDim+1;
             offset2=(l-1)*poolDim+1;
             temp(k,l)=sum(sum(cf(offset1:k*poolDim,offset2:l*poolDim)))/(poolDim*poolDim);
             end
        end
        pooledFeatures(j,i,:,:)=temp;
    end
end
end

