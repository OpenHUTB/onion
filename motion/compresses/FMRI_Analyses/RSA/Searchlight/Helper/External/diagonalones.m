function X = diagonalones(X,num);

if nargin < 2
    num = 1;
end

X(1:size(X,1)+1:end) = num;
