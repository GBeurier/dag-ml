function result = cvRefitPredict(dsl, controllers, envelope, adapter, varargin)
%CVREFITPREDICT Run CV, REFIT and PREDICT in one native DAG-ML session.
%   RESULT includes the bundle, OOF averages, replay prediction blocks and
%   native scores. Host model handles remain live only in this CLI session.
%   Options: 'cli', 'output', 'processWorkers', 'processTimeoutMs',
%   'processRetries', 'bundleId', 'variantId', 'selectionMetric',
%   'selections', 'planId', 'runId', 'rootSeed', 'scheduler',
%   'schedulerWorkers', 'cpuThreads', 'gpuDevices'.
if ispc
    error('dagml:CvRefitPredict:Platform', 'cvRefitPredict requires a POSIX shell.');
end
parser = inputParser;
addParameter(parser, 'cli', 'dag-ml-cli');
addParameter(parser, 'output', '');
addParameter(parser, 'processWorkers', 1);
addParameter(parser, 'processTimeoutMs', 30000);
addParameter(parser, 'processRetries', 0);
addParameter(parser, 'bundleId', 'bundle:cli.process.dsl.cv.refit.replay');
addParameter(parser, 'variantId', '');
addParameter(parser, 'selectionMetric', 'rmse');
addParameter(parser, 'selections', '');
addParameter(parser, 'planId', 'plan:cli.process.dsl.cv.refit.replay');
addParameter(parser, 'runId', 'run:cli.process.dsl.cv.refit.replay');
addParameter(parser, 'rootSeed', 12345);
addParameter(parser, 'scheduler', 'sequential');
addParameter(parser, 'schedulerWorkers', 1);
addParameter(parser, 'cpuThreads', 1);
addParameter(parser, 'gpuDevices', {});
parse(parser, varargin{:});
options = parser.Results;
inputs = {'--dsl', requiredFile(dsl, 'dsl'), ...
    '--controllers', requiredFile(controllers, 'controllers'), ...
    '--envelope', requiredFile(envelope, 'envelope'), ...
    '--adapter', requiredFile(adapter, 'adapter')};
metric = scalarText(options.selectionMetric, 'selection metric');
if ~any(strcmp(metric, {'rmse', 'accuracy', 'balanced_accuracy'}))
    error('dagml:CvRefitPredict:Option', ...
        'selectionMetric must be rmse, accuracy or balanced_accuracy.');
end
scheduler = scalarText(options.scheduler, 'scheduler');
if ~any(strcmp(scheduler, {'sequential', 'parallel'}))
    error('dagml:CvRefitPredict:Option', 'scheduler must be sequential or parallel.');
end
gpuDevices = options.gpuDevices;
if ischar(gpuDevices) || (isstring(gpuDevices) && isscalar(gpuDevices))
    gpuDevices = {char(gpuDevices)};
elseif isstring(gpuDevices)
    gpuDevices = cellstr(gpuDevices);
end
if ~iscell(gpuDevices)
    error('dagml:CvRefitPredict:Option', 'gpuDevices must contain text identifiers.');
end
arguments = {scalarText(options.cli, 'CLI'), 'run-process-dsl-cv-refit-replay'};
arguments = [arguments, inputs, ...
    {'--bundle-id', scalarText(options.bundleId, 'bundle ID'), ...
    '--plan-id', scalarText(options.planId, 'plan ID'), ...
    '--run-id', scalarText(options.runId, 'run ID'), ...
    '--selection-metric', metric, ...
    '--root-seed', integerOption(options.rootSeed, 'rootSeed', 0), ...
    '--scheduler', scheduler, ...
    '--scheduler-workers', integerOption(options.schedulerWorkers, 'schedulerWorkers', 1), ...
    '--cpu-threads', integerOption(options.cpuThreads, 'cpuThreads', 1), ...
    '--process-workers', integerOption(options.processWorkers, 'processWorkers', 1), ...
    '--process-timeout-ms', integerOption(options.processTimeoutMs, 'processTimeoutMs', 1), ...
    '--process-retries', integerOption(options.processRetries, 'processRetries', 0)}];
for index = 1:numel(gpuDevices)
    arguments(end + 1:end + 2) = ...
        {'--gpu-device', scalarText(gpuDevices{index}, 'GPU device')};
end
if ~isempty(options.variantId)
    arguments(end + 1:end + 2) = ...
        {'--variant-id', scalarText(options.variantId, 'variant ID')};
end
if ~isempty(options.selections)
    arguments(end + 1:end + 2) = ...
        {'--selections', requiredFile(options.selections, 'selections')};
end
temporaryOutput = isempty(options.output);
if temporaryOutput
    output = [tempname(), '.json'];
    cleanup = onCleanup(@() deleteIfExists(output)); %#ok<NASGU>
else
    output = scalarText(options.output, 'output');
end
arguments(end + 1:end + 2) = {'--output', output};
quoted = cellfun(@shellQuote, arguments, 'UniformOutput', false);
[status, message] = system(strjoin(quoted, ' '));
if status ~= 0
    error('dagml:CvRefitPredict:CLI', ...
        'dag-ml CV+REFIT+PREDICT failed (exit %d): %s', status, strtrim(message));
end
if exist(output, 'file') ~= 2
    error('dagml:CvRefitPredict:MissingOutcome', ...
        'dag-ml CV+REFIT+PREDICT exited without writing an outcome.');
end
result = jsondecode(fileread(output));
if ~isfield(result, 'bundle') || ~isfield(result, 'replay_node_results') || ...
        ~isfield(result, 'replay_prediction_blocks')
    error('dagml:CvRefitPredict:MissingEvidence', ...
        'dag-ml outcome lacks bundle or replay evidence.');
end
end

function path = requiredFile(value, label)
path = scalarText(value, label);
if exist(path, 'file') ~= 2
    error('dagml:CvRefitPredict:MissingFile', '%s does not exist: %s', label, path);
end
end

function text = scalarText(value, label)
if ischar(value) && isrow(value)
    text = value;
elseif isstring(value) && isscalar(value)
    text = char(value);
else
    error('dagml:CvRefitPredict:Option', '%s must be scalar text.', label);
end
if isempty(strtrim(text)) || any(text == char(0)) || ...
        any(text == char(10)) || any(text == char(13))
    error('dagml:CvRefitPredict:Option', ...
        '%s must be non-empty text without control characters.', label);
end
end

function text = integerOption(value, label, minimum)
if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value) || ...
        value < minimum || value ~= fix(value)
    error('dagml:CvRefitPredict:Option', '%s must be an integer >= %d.', label, minimum);
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
