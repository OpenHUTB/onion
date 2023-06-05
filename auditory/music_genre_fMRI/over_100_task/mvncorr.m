function predcc = mvncorr(mu1, mu2)
%
% Calculate Pearson's correlation coefficients for all cortical voxels
%

mu1_sub = mu1- ones(size(mu1,1),1)*nanmean(mu1);
mu2_sub = mu2- ones(size(mu2,1),1)*nanmean(mu2);
COV = nanmean(mu1_sub .* mu2_sub);

predcc = COV ./ sqrt( nanmean(mu1_sub.^2) .* nanmean(mu2_sub.^2));
predcc = predcc';

    