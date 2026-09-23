function result = hostHpoSearch( ...
    plan, envelope, request, operatorAdapter, optimizerAdapter, varargin)
%HOSTHPOSEARCH Run DAG-ML host hyperparameter search through the native CLI.
%   RESULT = dagml.hostHpoSearch(PLAN, ENVELOPE, REQUEST, OPERATORADAPTER,
%   OPTIMIZERADAPTER) reads the three JSON contracts from files. The two
%   executable adapters speak the DAG-ML JSONL operator/optimizer protocols.
%   Name-value options: 'cli', 'checkpoint', 'output',
%   'operatorPersistent', 'parallelTrials', and 'adapterTimeoutMs'.
%   DAG-ML starts an isolated operator-adapter process for each parallel trial.

if ispc
    error('dagml:HostHpo:Platform', ...
        'hostHpoSearch currently requires a POSIX shell.');
end
parser = inputParser;
addParameter(parser, 'cli', 'dag-ml-cli');
addParameter(parser, 'checkpoint', '');
addParameter(parser, 'output', '');
addParameter(parser, 'operatorPersistent', false);
addParameter(parser, 'parallelTrials', 1);
addParameter(parser, 'adapterTimeoutMs', 30000);
parse(parser, varargin{:});
options = parser.Results;

plan = scalarText(plan, 'plan');
envelope = scalarText(envelope, 'envelope');
request = scalarText(request, 'request');
operatorAdapter = scalarText(operatorAdapter, 'operator adapter');
optimizerAdapter = scalarText(optimizerAdapter, 'optimizer adapter');
cli = scalarText(options.cli, 'CLI');
required = {plan, envelope, request, operatorAdapter, optimizerAdapter};
for index = 1:numel(required)
    if exist(required{index}, 'file') ~= 2
        error('dagml:HostHpo:MissingFile', ...
            'Required input file does not exist: %s', required{index});
    end
end
if ~isnumeric(options.parallelTrials) || ~isscalar(options.parallelTrials) || ...
        ~isfinite(options.parallelTrials) || options.parallelTrials < 1 || ...
        fix(options.parallelTrials) ~= options.parallelTrials
    error('dagml:HostHpo:ParallelTrials', ...
        'parallelTrials must be a positive integer.');
end
if ~isnumeric(options.adapterTimeoutMs) || ...
        ~isscalar(options.adapterTimeoutMs) || ...
        ~isfinite(options.adapterTimeoutMs) || ...
        options.adapterTimeoutMs < 1 || ...
        fix(options.adapterTimeoutMs) ~= options.adapterTimeoutMs
    error('dagml:HostHpo:Timeout', ...
        'adapterTimeoutMs must be a positive integer.');
end
if ~islogical(options.operatorPersistent) || ...
        ~isscalar(options.operatorPersistent)
    error('dagml:HostHpo:Persistent', ...
        'operatorPersistent must be a logical scalar.');
end

temporaryOutput = isempty(options.output);
if temporaryOutput
    output = [tempname(), '.json'];
    cleanup = onCleanup(@() deleteIfExists(output)); %#ok<NASGU>
else
    output = scalarText(options.output, 'output');
end
arguments = {cli, 'run-host-hpo', '--plan', plan, ...
    '--envelope', envelope, '--request', request, ...
    '--operator-adapter', operatorAdapter, ...
    '--optimizer-adapter', optimizerAdapter, ...
    '--parallel-trials', sprintf('%.0f', options.parallelTrials), ...
    '--adapter-timeout-ms', sprintf('%.0f', options.adapterTimeoutMs), ...
    '--output', output};
if options.operatorPersistent
    arguments{end + 1} = '--operator-persistent';
end
if ~isempty(options.checkpoint)
    arguments(end + 1:end + 2) = ...
        {'--checkpoint', scalarText(options.checkpoint, 'checkpoint')};
end
quoted = cellfun(@shellQuote, arguments, 'UniformOutput', false);
[status, message] = system(strjoin(quoted, ' '));
if status ~= 0
    error('dagml:HostHpo:CLI', ...
        'dag-ml host HPO failed (exit %d): %s', status, strtrim(message));
end
if exist(output, 'file') ~= 2
    error('dagml:HostHpo:MissingResult', ...
        'dag-ml host HPO exited without writing a result.');
end
result = jsondecode(fileread(output));
end

function text = scalarText(value, label)
if ischar(value) && (isrow(value) || isempty(value))
    text = value;
elseif isstring(value) && isscalar(value)
    text = char(value);
else
    error('dagml:HostHpo:InvalidText', ...
        '%s must be scalar text.', label);
end
if isempty(strtrim(text)) || any(text == char(0)) || ...
        any(text == char(10)) || any(text == char(13))
    error('dagml:HostHpo:InvalidText', ...
        '%s must be non-empty text without NUL or newlines.', label);
end
end

function quoted = shellQuote(value)
quote = char(39);
quoted = [quote, strrep(value, quote, [quote, '"', quote, '"', quote]), quote];
end

function deleteIfExists(path)
if exist(path, 'file') == 2
    delete(path);
end
end
