function result = initialFullRefit( ...
    dsl, controllers, envelope, trainingSampleIds, adapter, packageOutput, varargin)
%INITIALFULLREFIT Capture a signed no-CV full REFIT package with DAG-ML.
%   The host keeps model sidecar bytes; the package records their identities.
%   Options: 'cli', 'output', 'persistent', 'processWorkers',
%   'processTimeoutMs', 'processRetries', 'packageId', 'planId', 'runId',
%   'rootSeed', 'scheduler', 'schedulerWorkers', 'cpuThreads', 'gpuDevices'.
if ispc
    error('dagml:InitialRefit:Platform', 'initialFullRefit requires a POSIX shell.');
end
parser = inputParser;
addParameter(parser, 'cli', 'dag-ml-cli');
addParameter(parser, 'output', '');
addParameter(parser, 'persistent', false);
addParameter(parser, 'processWorkers', 1);
addParameter(parser, 'processTimeoutMs', 30000);
addParameter(parser, 'processRetries', 0);
addParameter(parser, 'packageId', 'package:cli.process.dsl.initial.refit');
addParameter(parser, 'planId', 'plan:cli.process.dsl.refit.phase');
addParameter(parser, 'runId', 'run:cli.process.dsl.refit.phase');
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
    '--training-sample-ids', requiredFile(trainingSampleIds, 'training sample IDs'), ...
    '--adapter', requiredFile(adapter, 'adapter')};
packageOutput = scalarText(packageOutput, 'package output');
schedule = scalarText(options.scheduler, 'scheduler');
if ~any(strcmp(schedule, {'sequential', 'parallel'}))
    error('dagml:InitialRefit:Option', 'scheduler must be sequential or parallel.');
end
gpuDevices = options.gpuDevices;
if ischar(gpuDevices) || (isstring(gpuDevices) && isscalar(gpuDevices))
    gpuDevices = {char(gpuDevices)};
elseif isstring(gpuDevices)
    gpuDevices = cellstr(gpuDevices);
end
if ~iscell(gpuDevices)
    error('dagml:InitialRefit:Option', 'gpuDevices must contain text identifiers.');
end
extras = {'--package-output', packageOutput, ...
    '--package-id', scalarText(options.packageId, 'package ID'), ...
    '--plan-id', scalarText(options.planId, 'plan ID'), ...
    '--run-id', scalarText(options.runId, 'run ID'), ...
    '--root-seed', integerOption(options.rootSeed, 'rootSeed', 0), ...
    '--scheduler', schedule, ...
    '--scheduler-workers', integerOption(options.schedulerWorkers, 'schedulerWorkers', 1), ...
    '--cpu-threads', integerOption(options.cpuThreads, 'cpuThreads', 1)};
for index = 1:numel(gpuDevices)
    extras(end + 1:end + 2) = ...
        {'--gpu-device', scalarText(gpuDevices{index}, 'GPU device')};
end
result = runCli('run-process-dsl-refit-phase', inputs, ...
    extras, options);
if exist(packageOutput, 'file') ~= 2
    error('dagml:InitialRefit:MissingPackage', ...
        'dag-ml REFIT exited without writing its package.');
end
if ~isfield(result, 'initial_full_refit_package')
    error('dagml:InitialRefit:MissingPackage', ...
        'dag-ml REFIT outcome lacks initial_full_refit_package.');
end
end

function result = runCli(command, inputs, extras, options)
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
cliArgs = [{cli, command}, inputs, extras, ...
    {'--process-workers', workers, '--process-timeout-ms', timeout, ...
    '--process-retries', retries, '--output', output}];
if options.persistent
    cliArgs{end + 1} = '--persistent';
end
quoted = cellfun(@shellQuote, cliArgs, 'UniformOutput', false);
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
