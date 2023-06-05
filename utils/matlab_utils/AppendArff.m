% function AppendArff:
%
% Loads the given ARFF file appends the provided attribute and saves the data
% the original file
%
% input:
%   arffFile    - file to load the data from
%   attData     - attribute data
%   attName     - attribute name
%   attType     - attribute type (Integer or Numeric accepted for now)
%   extraId     - extension to add to the name before file extension

function AppendArff(arffFile, attData, attName, attType, extraId)
    if (nargin<5)
        extraId = '';
    end
    % initially load data
    [data, metadata, attributes, relation] = LoadArff(arffFile);

    % check if attribute already exists
    for i=1:size(attributes,1)
        if (strcmpi(attributes{i,1}, attName))
            error(['Attributes "' attName '" already exists in file ' arffFile]);
        end
    end

    % check for data and attribute values
    assert(size(data,1)==size(attData,1), 'Attribute data and arff data should be of the same length');

    % set data
    appData = zeros(size(data,1), size(data,2)+1);
    appData(:,1:end-1) = data;
    appData(:,end) = attData;

    % append attribute
    index = size(attributes,1)+1;
    attributes{index,1} = attName;
    attributes{index,2} = attType;

    [dir, name, ext] = fileparts(arffFile);
    outArffFile = [name extraId ext];
    if (~isempty(dir))
        outArffFile = [dir '/' outArffFile];
    end

    % save to the same file
    SaveArff(outArffFile, appData, metadata, attributes, relation);
end    
