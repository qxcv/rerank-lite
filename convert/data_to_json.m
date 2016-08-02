function data_to_json(data_path, output_path)
%DATA_TO_JSON Handy script to convert .mat files to JSON (e.g. MPII data)
% Uses jsonlab internally (included)
if ~exist('savejson', 'var')
    addpath('convert/jsonlab');
end

fprintf('Loading from %s\n', data_path);
loaded = load(data_path);
names = fieldnames(loaded);
if length(names) ~= 1
    error('data_to_json:loadfail', ...
        'Expected one key in loaded file, got %i', length(names));
end
name = names{1};

fprintf('Saving to %s\n', output_path);
savejson(name, loaded.(name), output_path);
fprintf('Done\n');
end

