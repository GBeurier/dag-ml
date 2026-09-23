function octave_ridge_oracle()
% Host-owned Ridge operator. DAG-ML owns fold identity, scoring and artifacts.
if strcmp(getenv('DAGML_OCTAVE_MODE'), 'describe')
    description = struct('schema_version', 1, 'protocol', 'dag-ml-process-adapter', ...
        'adapter_id', 'dag-ml-octave-ridge-oracle', ...
        'supported_modes', {{'one_shot', 'jsonl'}}, ...
        'capabilities', {{'control_frames_v1', 'node_task_json_v1', ...
            'node_result_json_v1', 'parallel_invocation_v1', ...
            'persistent_workers', 'worker_env', 'stateful_refit_artifacts'}});
    fprintf(1, '%s\n', jsonencode(description));
    return;
end
rows = read_rows(getenv('DAGML_OCTAVE_HPO_DATA'));
while true
    line = read_json_line();
    if ~ischar(line), break; end
    frame = jsondecode(line);
    if isfield(frame, 'type')
        switch char(frame.type)
            case 'init'
                fprintf(1, '{"type":"ack","schema_version":1,"status":"initialized"}\n');
                fflush(1);
                continue;
            case 'close'
                fprintf(1, '{"type":"ack","schema_version":1,"status":"closed"}\n');
                fflush(1);
                break;
            case 'task'
                task = frame.task;
            otherwise
                error('Octave Ridge received an unsupported control frame');
        end
    else
        task = frame;
    end
    result = run_task(task, line, rows);
    if isfield(frame, 'type')
        encoded = jsonencode(struct('type', 'result', 'schema_version', 1, 'result', result));
    else
        encoded = jsonencode(result);
    end
    if strcmp(char(task.phase), 'REFIT')
        artifact_id = ['artifact:' char(task.node_plan.node_id) ':octave-ridge:refit'];
        handle = struct('handle', 7001, 'kind', 'model', ...
            'owner_controller', char(task.node_plan.controller_id));
        handle_map = ['"artifact_handles":{' jsonencode(artifact_id) ':' ...
            jsonencode(handle) '}'];
        encoded = strrep(encoded, '"artifact_handles":{}', handle_map);
    end
    seed_tokens = regexp(line, '"seed"\s*:\s*([0-9]+)', 'tokens');
    assert(~isempty(seed_tokens));
    seed = seed_tokens{end}{1};
    encoded = strrep(encoded, '"seed":"__DAGML_U64_SEED__"', ['"seed":' seed]);
    fprintf(1, '%s\n', encoded);
    fflush(1);
end
end

function line = read_json_line()
% fgetl(0) waits for pipe EOF on Octave when DAG-ML keeps the worker alive.
line = '';
while true
    byte = fread(0, 1, 'char=>char');
    if isempty(byte)
        if isempty(line), line = -1; end
        return;
    end
    if byte == char(10), return; end
    line(end + 1) = byte;
end
end

function rows = read_rows(path)
fid = fopen(path, 'r');
assert(fid >= 0, 'Octave Ridge data file is missing');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
assert(ischar(fgetl(fid)));
fields = textscan(fid, '%s%f%f', 'Delimiter', ',', 'CollectOutput', false);
rows = struct('id', {fields{1}}, 'x', fields{2}, 'y', fields{3});
assert(numel(rows.id) == numel(unique(rows.id)));
end

function indices = select_rows(rows, ids)
ids = cellstr(ids);
assert(~isempty(ids) && numel(ids) == numel(unique(ids)));
[present, indices] = ismember(ids, rows.id);
assert(all(present), 'Octave Ridge received a sample absent from its data table');
end

function result = run_task(task, raw_line, rows)
phase = char(task.phase);
node = char(task.node_plan.node_id);
controller = char(task.node_plan.controller_id);
variant = char(task.variant_id);
alpha = double(task.node_plan.params.n_components) - 1;
assert(isfinite(alpha) && alpha >= 0);
artifact_id = ['artifact:' node ':octave-ridge:refit'];
sidecar = getenv('DAGML_OCTAVE_RIDGE_SIDECAR');
artifacts = {};
artifact_handles = struct();
targets = {};
if strcmp(phase, 'FIT_CV')
    plan = jsondecode(fileread(getenv('DAGML_OCTAVE_HPO_PLAN')));
    folds = as_cell(plan.fold_set.folds);
    fold_id = char(task.fold_id);
    matches = cellfun(@(fold) strcmp(char(fold.fold_id), fold_id), folds);
    assert(sum(matches) == 1);
    fold = folds{find(matches, 1)};
    train_ids = cellstr(fold.train_sample_ids);
    ids = cellstr(fold.validation_sample_ids);
    assert(isempty(intersect(train_ids, ids)));
    train = select_rows(rows, train_ids);
    selected = select_rows(rows, ids);
    weight = sum(rows.x(train) .* rows.y(train)) / ...
        (sum(rows.x(train) .* rows.x(train)) + alpha);
    predictions = weight .* rows.x(selected);
    partition = 'validation';
    targets = {struct('level', 'sample', ...
        'unit_ids', {cellfun(@(id) struct('level', 'sample', 'id', id), ...
            ids, 'UniformOutput', false)}, ...
        'values', {cellfun(@(value) {value}, num2cell(rows.y(selected)), ...
            'UniformOutput', false)}, 'target_names', {{'y'}})};
    evidence_dir = getenv('DAGML_OCTAVE_HPO_EVIDENCE_DIR');
    if ~isempty(evidence_dir)
        evidence = struct('train_ids', {train_ids}, 'validation_ids', {ids}, ...
            'predictions', {num2cell(predictions)}, 'targets', {num2cell(rows.y(selected))}, ...
            'alpha', alpha);
        filename = [strrep(variant, ':', '_') '_' strrep(fold_id, ':', '_') '.json'];
        write_text(fullfile(evidence_dir, filename), jsonencode(evidence));
    end
elseif strcmp(phase, 'REFIT')
    ids = view_ids(task, 'full_train');
    selected = select_rows(rows, ids);
    model = struct('weight', sum(rows.x(selected) .* rows.y(selected)) / ...
        (sum(rows.x(selected) .* rows.x(selected)) + alpha), ...
        'alpha', alpha, 'train_ids', {ids});
    [parent, ~, ~] = fileparts(sidecar);
    if ~exist(parent, 'dir'), mkdir(parent); end
    save('-mat7-binary', sidecar, 'model');
    artifact = struct('id', artifact_id, 'kind', 'octave_ridge_model', ...
        'controller_id', controller, 'backend', 'mat', ...
        'uri', 'artifacts/octave-ridge.mat', ...
        'content_fingerprint', file_sha256(sidecar), ...
        'size_bytes', dir(sidecar).bytes, ...
        'plugin', 'dagml.octave_ridge_oracle', 'plugin_version', '1.0.0');
    artifacts = {artifact};
    predictions = model.weight .* rows.x(selected);
    partition = 'final';
    targets = {struct('level', 'sample', ...
        'unit_ids', {cellfun(@(id) struct('level', 'sample', 'id', id), ...
            ids, 'UniformOutput', false)}, ...
        'values', {cellfun(@(value) {value}, num2cell(rows.y(selected)), ...
            'UniformOutput', false)}, 'target_names', {{'y'}})};
elseif strcmp(phase, 'PREDICT')
    handles = as_cell(task.input_handles);
    handles = handles(cellfun(@(item) strcmp(char(item.kind), 'model'), handles));
    assert(numel(handles) == 1);
    inputs = as_cell(task.artifact_inputs);
    assert(numel(inputs) == 1);
    input = inputs{1};
    assert(strcmp(char(input.node_id), node) && strcmp(char(input.controller_id), controller));
    assert(strcmp(char(input.artifact.id), artifact_id));
    assert(strcmp(char(input.artifact.backend), 'mat'));
    assert(strcmp(char(input.artifact.uri), 'artifacts/octave-ridge.mat'));
    assert(strcmp(char(input.artifact.content_fingerprint), file_sha256(sidecar)));
    loaded = load(sidecar, 'model');
    model = loaded.model;
    assert(model.alpha == alpha);
    ids = view_ids(task, 'predict');
    selected = select_rows(rows, ids);
    predictions = model.weight .* rows.x(selected);
    partition = 'final';
else
    error('Octave Ridge received an unsupported phase');
end
prediction = struct('producer_node', node, 'partition', partition, ...
    'fold_id', NaN, 'sample_ids', {ids}, ...
    'values', {cellfun(@(value) {value}, num2cell(predictions), ...
        'UniformOutput', false)}, 'target_names', {{'y'}});
if strcmp(phase, 'FIT_CV'), prediction.fold_id = fold_id; end
lineage_id = ['lineage:octave-ridge:' phase ':' variant];
if strcmp(phase, 'FIT_CV'), lineage_id = [lineage_id ':' fold_id]; end
lineage_fold_id = task.fold_id;
if isempty(lineage_fold_id), lineage_fold_id = NaN; end
lineage = struct('record_id', lineage_id, ...
    'run_id', task.run_id, 'node_id', node, 'phase', phase, ...
    'controller_id', controller, ...
    'controller_version', task.node_plan.controller_version, ...
    'variant_id', variant, 'fold_id', lineage_fold_id, ...
    'branch_path', {task.branch_path}, 'input_lineage', {{}}, ...
    'artifact_refs', {artifacts}, ...
    'params_fingerprint', task.node_plan.params_fingerprint, ...
    'data_model_shape_fingerprint', NaN, ...
    'aggregation_policy_fingerprint', NaN, ...
    'seed', '__DAGML_U64_SEED__', 'unsafe_flags', {{}}, ...
    'metrics', struct(), 'loss_attestations', {{}}, ...
    'early_stopping_records', {{}});
result = struct('node_id', node, 'outputs', struct(), ...
    'predictions', {{prediction}}, 'lineage', lineage);
if ~isempty(targets), result.regression_targets = targets; end
if strcmp(phase, 'REFIT')
    result.outputs = struct('oof', struct('handle', 7002, ...
        'kind', 'prediction', 'owner_controller', controller));
    result.artifacts = artifacts;
    % MATLAB fields cannot contain colons; encode this exact map above.
    result.artifact_handles = struct();
elseif strcmp(phase, 'PREDICT')
    result.outputs = struct('oof', struct('handle', 7002, ...
        'kind', 'prediction', 'owner_controller', controller));
    result.artifacts = {};
    result.artifact_handles = struct();
end
end

function ids = view_ids(task, partition)
views = as_cell(task.data_views);
matches = cellfun(@(view) strcmp(char(view.partition), partition), views);
assert(sum(matches) == 1);
ids = cellstr(views{find(matches, 1)}.sample_ids);
assert(~isempty(ids));
end

function values = as_cell(value)
if iscell(value)
    values = value;
elseif isstruct(value) && isscalar(value) && ...
        ~isfield(value, 'partition') && ~isfield(value, 'fold_id') && ...
        ~isfield(value, 'kind') && ~isfield(value, 'artifact')
    values = struct2cell(value);
else
    values = num2cell(value);
end
end

function fingerprint = file_sha256(path)
fingerprint = hash('sha256', fileread(path));
end

function write_text(path, content)
fid = fopen(path, 'w');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', content);
end
