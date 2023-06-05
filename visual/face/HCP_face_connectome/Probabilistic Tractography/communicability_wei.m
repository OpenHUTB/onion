function   F = communicability_wei(CIJ)

%%% this function converts structural connectivity matrix to communicability matrix
%%% the diagnal of structural connectivity matrix must be zeros (not NaN)

%inputs
%           CIJ    weighted connection matrix
%           i      row
%           j      column
%
%outputs
%           F      communicability
%=================================================

N = size(CIJ,1);

B = sum(CIJ')';
C = diag(B);
D = C^(-(1/2));
E = D * CIJ * D;
F = expm(E);
F = F.*~eye(N);

