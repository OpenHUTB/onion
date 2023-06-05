function SSinds = getResampInd_repN(samplenum, repN , divsample)

if ~exist('divsample', 'var')
    divnum = 50;
else
    divnum = samplenum/divsample;
end

% get cross validation samples
cvFrac = 0.2;
randSeed = 1234;
rand('twister',randSeed);
randn('state',randSeed);

zs=floor(samplenum/divnum);
zr = reshape(1:zs*divnum, zs, divnum);

for nn = 1:repN

    ssinds = [];
    for ii=1:10
      a=randperm(divnum);
      regInd = zr(:,a(1:round(divnum*cvFrac)));
      regInd = regInd(:)';
      trnInd = setdiff(1:samplenum, [regInd]);
      ssinds(ii).regInd = regInd;
      ssinds(ii).trnInd = trnInd;
    end

    SSinds{nn} = ssinds;

end