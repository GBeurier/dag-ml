function results = executeExecutionPlanPhase( ...
    executionPlan, trustedControllerManifests, runId, rootSeed, phase, ...
    controllers, nativeLibrary)
% Execute one ExecutionPlan phase through the native DAG-ML scheduler.

if nargin < 7
    nativeLibrary = getenv('DAGML_NATIVE_LIBRARY');
end
runId = scalarText(runId, 'run id');
phase = scalarText(phase, 'phase');
if ~any(strcmp(phase, {'FIT_CV', 'SELECT', 'REFIT', 'PREDICT', 'EXPLAIN'}))
    error('dagml:ExecutePhase:InvalidPhase', ...
        'Phase must be FIT_CV, SELECT, REFIT, PREDICT, or EXPLAIN.');
end
if ~isnumeric(rootSeed) || ~isscalar(rootSeed) || ~isfinite(rootSeed) || ...
        fix(rootSeed) ~= rootSeed || rootSeed < 0 || rootSeed > flintmax
    error('dagml:ExecutePhase:RootSeed', ...
        'rootSeed must be a non-negative safe integer.');
end
nativeLibrary = scalarText(nativeLibrary, 'native library');

[controllerIds, rawCallbacks] = controllerCallbacks(controllers);
callbacks = cell(size(rawCallbacks));
for index = 1:numel(rawCallbacks)
    callback = rawCallbacks{index};
    callbacks{index} = @(controllerId, taskJson) nodeResultJson( ...
        callback(controllerId, taskJson), taskJson);
end

resultJson = dagml.executeExecutionPlanPhaseNative( ...
    jsonText(executionPlan), ...
    jsonText(trustedControllerManifests), ...
    runId, double(rootSeed), phase, controllerIds, callbacks, nativeLibrary);
results = jsondecode(resultJson);
end

function text = jsonText(value)
if ischar(value) && (isrow(value) || isempty(value))
    text = value;
elseif isstring(value) && isscalar(value)
    text = char(value);
else
    text = jsonencode(value);
end
end

function text = scalarText(value, label)
if ischar(value) && (isrow(value) || isempty(value))
    text = value;
elseif isstring(value) && isscalar(value)
    text = char(value);
else
    error('dagml:ExecutePhase:InvalidText', ...
        '%s must be scalar text.', label);
end
if isempty(strtrim(text))
    error('dagml:ExecutePhase:InvalidText', ...
        '%s must be non-empty text.', label);
end
end

function [ids, callbacks] = controllerCallbacks(controllers)
if isa(controllers, 'containers.Map')
    ids = keys(controllers);
    callbacks = values(controllers, ids);
elseif iscell(controllers) && size(controllers, 2) == 2
    ids = controllers(:, 1)';
    callbacks = controllers(:, 2)';
else
    error('dagml:ExecutePhase:Controllers', ...
        'controllers must be a containers.Map or a two-column cell array.');
end
for index = 1:numel(ids)
    ids{index} = scalarText(ids{index}, 'controller id');
    if ~isa(callbacks{index}, 'function_handle')
        error('dagml:ExecutePhase:Controllers', ...
            'controller callbacks must be function handles.');
    end
end
if numel(unique(ids)) ~= numel(ids)
    error('dagml:ExecutePhase:Controllers', ...
        'controllers must be keyed by unique controller ids.');
end
end

function resultJson = nodeResultJson(value, taskJson)
task = jsondecode(taskJson);
if ischar(value) || (isstring(value) && isscalar(value))
    result = jsondecode(char(value));
else
    result = value;
end
if ~isstruct(result) || ~isscalar(result) || ~isfield(result, 'lineage')
    error('dagml:ExecutePhase:NodeResult', ...
        'controller callback must return a NodeResult with lineage.');
end
if ~isfield(result.lineage, 'seed') || isempty(result.lineage.seed)
    if isfield(task, 'seed')
        result.lineage.seed = task.seed;
    else
        result.lineage.seed = [];
    end
end
resultJson = jsonencode(result);
end
