function RDM = flipandfold(RDM);

for i = 1:size(RDM,1);
    for j = 1:size(RDM,2);
        v = [RDM(i,j) RDM(j,i)];
        new_RDM(i,j) = mean(v);
        new_RDM(j,i) = mean(v);
    end
end

RDM = diagonalones(new_RDM,0);