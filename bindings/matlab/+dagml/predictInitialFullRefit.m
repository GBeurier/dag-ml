function result = predictInitialFullRefit( ...
    package, envelope, adapter, artifactHandles, outputIds, varargin)
%PREDICTINITIALFULLREFIT Replay PREDICT from a signed no-CV REFIT package.
%   A separate V2 prediction cohort and the exact host-sidecar artifact
%   handles are required. Options: 'cli', 'output', 'persistent',
%   'processWorkers', 'processTimeoutMs', 'processRetries', 'runId'.
if ispc
    error('dagml:InitialRefit:Platform', 'predictInitialFullRefit requires a POSIX shell.');
end
parser = inputParser;
addParameter(parser, 'cli', 'dag-ml-cli');
addParameter(parser, 'output', '');
addParameter(parser, 'persistent', false);
addParameter(parser, 'processWorkers', 1);
addParameter(parser, 'processTimeoutMs', 30000);
addParameter(parser, 'processRetries', 0);
addParameter(parser, 'runId', 'run:cli.initial.refit.predict');
parse(parser, varargin{:});
options = parser.Results;
inputs = {'--package', requiredFile(package, 'package'), ...
    '--envelope', requiredFile(envelope, 'envelope'), ...
    '--adapter', requiredFile(adapter, 'adapter'), ...
    '--artifact-handles', requiredFile(artifactHandles, 'artifact handles'), ...
    '--output-ids', requiredFile(outputIds, 'output IDs')};
result = runCli('run-process-initial-full-refit-predict', inputs, options);
if ~isfield(result, 'replay_outcome')
    error('dagml:InitialRefit:MissingReplay', ...
        'dag-ml PREDICT outcome lacks replay_outcome.');
end
end

function result = runCli(command, inputs, options)
cli = scalarText(options.cli, 'CLI');
workers = integerOption(options.processWorkers, 'processWorkers', 1);
timeout = integerOption(options.processTimeoutMs, 'processTimeoutMs', 1);
retries = integerOption(options.processRetries, 'processRetries', 0);
if ~islogical(options.persistent) || ~isscalar(options.persistent)
    error('dagml:InitialRefit:Option', 'persistent must be a logical scalar.');
end
temporaryOutput = isempty(options.output);
if temporaryOutput
    output = [tempname(), '.json'];
    cleanup = onCleanup(@() deleteIfExists(output)); %#ok<NASGU>
else
    output = scalarText(options.output, 'output');
end
arguments = [{cli, command}, inputs, ...
    {'--run-id', scalarText(options.runId, 'run ID')}, ...
    {'--process-workers', workers, '--process-timeout-ms', timeout, ...
    '--process-retries', retries, '--output', output}];
if options.persistent
    arguments{end + 1} = '--persistent';
end
quoted = cellfun(@shellQuote, arguments, 'UniformOutput', false);
[status, message] = system(strjoin(quoted, ' '));
if status ~= 0
    error('dagml:InitialRefit:CLI', 'dag-ml %s failed (exit %d): %s', ...
        command, status, strtrim(message));
end
if exist(output, 'file') ~= 2
    error('dagml:InitialRefit:MissingOutcome', ...
        'dag-ml %s exited without writing an outcome.', command);
end
result = jsondecode(fileread(output));
end

function path = requiredFile(value, label)
path = scalarText(value, label);
if exist(path, 'file') ~= 2
    error('dagml:InitialRefit:MissingFile', '%s does not exist: %s', label, path);
end
end

function text = scalarText(value, label)
if ischar(value) && isrow(value)
    text = value;
elseif isstring(value) && isscalar(value)
    text = char(value);
else
    error('dagml:InitialRefit:Option', '%s must be scalar text.', label);
end
if isempty(strtrim(text)) || any(text == char(0)) || ...
        any(text == char(10)) || any(text == char(13))
    error('dagml:InitialRefit:Option', ...
        '%s must be non-empty text without control characters.', label);
end
end

function text = integerOption(value, label, minimum)
if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value) || ...
        value < minimum || value ~= fix(value)
    error('dagml:InitialRefit:Option', '%s must be an integer >= %d.', label, minimum);
end
text = sprintf('%.0f', value);
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
