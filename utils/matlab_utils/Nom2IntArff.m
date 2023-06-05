% Nom2IntArff.m
%
% This function converts all the nominal attributes in the provided ARFF file
% to integer attributes. This helps speeding-up loading and saving times for
% very large ARFF files.
%
% input:
%   inputfile   - ARFF file to convert
%   outputfile  - file to store changed ARFF

function Nom2IntArff(inputfile, outputfile)
    [data, metadata, attributes, relation, comments] = LoadArff(inputfile);

    for i=1:size(attributes,1)
        isNom = IsNomAttribute(attributes{i,2});
        if (isNom)
            description = [' Attribute ' attributes{i,1} ' ' attributes{i,2} ' was converted to integer'];
            comments{end+1} = description;
            attributes{i,2} = 'integer';
        end
    end

    SaveArff(outputfile, data, metadata, attributes, relation, comments);
end
