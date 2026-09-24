function result = replayBundle( ...
    graph, campaign, controllers, bundle, replayRequest, adapter, ...
    artifactHandles, envelopes, varargin)
%REPLAYBUNDLE Replay a persisted CV bundle with host-owned model sidecars.
%   ENVELOPES is a containers.Map from replay data key to V2 envelope path.
%   ARTIFACTHANDLES is a JSON file of invocation-local handles, one per
%   bundle REFIT artifact. The host adapter resolves and verifies sidecars.
%   Options: 'cli', 'output', 'persistent', 'processWorkers',
%   'processTimeoutMs', 'processRetries', 'predictionCachePayload',
%   'predictionCacheStore', 'scoreOutput', 'planId', 'runId',
%   'rootSeed', 'scheduler', 'schedulerWorkers'.
if ispc
    error('dagml:ReplayBundle:Platform', 'replayBundle requires a POSIX shell.');
end
parser = inputParser;
addParameter(parser, 'cli', 'dag-ml-cli');
addParameter(parser, 'output', '');
addParameter(parser, 'persistentMode', false);
addParameter(parser, 'processWorkers', 1);
addParameter(parser, 'processTimeoutMs', 30000);
addParameter(parser, 'processRetries', 0);
addParameter(parser, 'predictionCachePayload', '');
addParameter(parser, 'predictionCacheStore', '');
addParameter(parser, 'scoreOutput', '');
addParameter(parser, 'planId', 'plan:cli.bundle');
addParameter(parser, 'runId', 'run:cli.process.replay');
addParameter(parser, 'rootSeed', 12345);
addParameter(parser, 'scheduler', 'sequential');
addParameter(parser, 'schedulerWorkers', 1);
for index = 1:2:numel(varargin)
    if strcmpi(varargin{index}, 'persistent')
        varargin{index} = 'persistentMode';
    end
end
parse(parser, varargin{:});
options = parser.Results;
if ~isa(envelopes, 'containers.Map') || envelopes.Count == 0
    error('dagml:ReplayBundle:Envelopes', ...
        'envelopes must be a non-empty containers.Map of keys to JSON paths.');
end
scheduler = scalarText(options.scheduler, 'scheduler');
if ~any(strcmp(scheduler, {'sequential', 'parallel'}))
    error('dagml:ReplayBundle:Option', 'scheduler must be sequential or parallel.');
end
if ~islogical(options.persistentMode) || ~isscalar(options.persistentMode)
    error('dagml:ReplayBundle:Option', 'persistent must be a logical scalar.');
end
cliArgs = {scalarText(options.cli, 'CLI'), 'run-process-replay', ...
    '--graph', requiredFile(graph, 'graph'), ...
    '--campaign', requiredFile(campaign, 'campaign'), ...
    '--controllers', requiredFile(controllers, 'controllers'), ...
    '--bundle', requiredFile(bundle, 'bundle'), ...
    '--replay-request', requiredFile(replayRequest, 'replay request'), ...
    '--adapter', requiredFile(adapter, 'adapter'), ...
    '--artifact-handles', requiredFile(artifactHandles, 'artifact handles'), ...
    '--plan-id', scalarText(options.planId, 'plan ID'), ...
    '--run-id', scalarText(options.runId, 'run ID'), ...
    '--root-seed', integerOption(options.rootSeed, 'rootSeed', 0), ...
    '--scheduler', scheduler, ...
    '--scheduler-workers', integerOption(options.schedulerWorkers, 'schedulerWorkers', 1), ...
    '--process-workers', integerOption(options.processWorkers, 'processWorkers', 1), ...
    '--process-timeout-ms', integerOption(options.processTimeoutMs, 'processTimeoutMs', 1), ...
    '--process-retries', integerOption(options.processRetries, 'processRetries', 0)};
keys = envelopes.keys();
for index = 1:numel(keys)
    key = scalarText(keys{index}, 'envelope key');
    if any(key == '=')
        error('dagml:ReplayBundle:Envelopes', 'envelope keys cannot contain =.');
    end
    path = requiredFile(envelopes(keys{index}), 'envelope');
    cliArgs(end + 1:end + 2) = {'--envelope', [key, '=', path]};
end
if ~isempty(options.predictionCachePayload)
    cliArgs(end + 1:end + 2) = ...
        {'--prediction-cache-payload', ...
        requiredFile(options.predictionCachePayload, 'prediction cache payload')};
end
if ~isempty(options.predictionCacheStore)
    cliArgs(end + 1:end + 2) = ...
        {'--prediction-cache-store', ...
        requiredFile(options.predictionCacheStore, 'prediction cache store')};
end
if ~isempty(options.scoreOutput)
    cliArgs(end + 1:end + 2) = ...
        {'--score-output', scalarText(options.scoreOutput, 'score output')};
end
if options.persistentMode
    cliArgs{end + 1} = '--persistent';
end
temporaryOutput = isempty(options.output);
if temporaryOutput
    output = [tempname(), '.json'];
    cleanup = onCleanup(@() deleteIfExists(output)); %#ok<NASGU>
else
    output = scalarText(options.output, 'output');
end
cliArgs(end + 1:end + 2) = {'--output', output};
quoted = cellfun(@shellQuote, cliArgs, 'UniformOutput', false);
[status, message] = system(strjoin(quoted, ' '));
if status ~= 0
    error('dagml:ReplayBundle:CLI', ...
        'dag-ml bundle replay failed (exit %d): %s', status, strtrim(message));
end
if exist(output, 'file') ~= 2
    error('dagml:ReplayBundle:MissingOutcome', ...
        'dag-ml bundle replay exited without writing an outcome.');
end
result = jsondecode(fileread(output));
if ~isfield(result, 'bundle_id') || ~isfield(result, 'node_results') || ...
        ~isfield(result, 'prediction_blocks')
    error('dagml:ReplayBundle:MissingEvidence', ...
        'dag-ml bundle replay outcome lacks native prediction evidence.');
end
end

function path = requiredFile(value, label)
path = scalarText(value, label);
if exist(path, 'file') ~= 2
    error('dagml:ReplayBundle:MissingFile', '%s does not exist: %s', label, path);
end
end

function text = scalarText(value, label)
if ischar(value) && isrow(value)
    text = value;
elseif isstring(value) && isscalar(value)
    text = char(value);
else
    error('dagml:ReplayBundle:Option', '%s must be scalar text.', label);
end
if isempty(strtrim(text)) || any(text == char(0)) || ...
        any(text == char(10)) || any(text == char(13))
    error('dagml:ReplayBundle:Option', ...
        '%s must be non-empty text without control characters.', label);
end
end

function text = integerOption(value, label, minimum)
if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value) || ...
        value < minimum || value ~= fix(value)
    error('dagml:ReplayBundle:Option', '%s must be an integer >= %d.', label, minimum);
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
