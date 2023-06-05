function ssinds = getResampInd(samplenum, divsample)

if ~exist('divsample', 'var')
    divnum = 50;
else
    divnum = samplenum/divsample;
end

% get cross validation samples
cvFrac = 0.1;
randSeed = 1234;
rand('twister',randSeed);
randn('state',randSeed);

zs=floor(samplenum/divnum);
zr = reshape(1:zs*divnum, zs, divnum);

ssinds = [];
for ii=1:10
  a=randperm(divnum);
  regInd = zr(:,a(1:round(divnum*cvFrac)));
  regInd = regInd(:)';
  trnInd = setdiff(1:samplenum, [regInd]);
  ssinds(ii).regInd = regInd;
  ssinds(ii).trnInd = trnInd;
end

