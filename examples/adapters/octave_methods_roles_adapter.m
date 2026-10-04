function octave_methods_roles_adapter()
% Persistent DAG-ML controller. Numerical recipes and states are Methods-owned.
parts = sscanf(version(), '%d.%d');
assert(numel(parts) >= 2 && (parts(1) > 7 || (parts(1) == 7 && parts(2) >= 1)) && ...
    exist('jsondecode', 'builtin') == 5 && exist('jsonencode', 'builtin') == 5, ...
    'The product adapter requires Octave 7.1+ with native JSON support');
if strcmp(getenv('DAGML_OCTAVE_MODE'), 'describe')
    fprintf(1, '%s\n', jsonencode(struct('schema_version', 1, ...
        'protocol', 'dag-ml-process-adapter', 'adapter_id', 'dag-ml-octave-methods-roles', ...
        'supported_modes', {{'jsonl'}}, 'capabilities', {{'control_frames_v1', ...
        'node_task_json_v1', 'node_result_json_v1', 'persistent_workers', ...
        'portable_artifact_payloads_v1', 'worker_env'}})));
    return;
end
methods_path = getenv('DAG_ML_OCTAVE_METHODS_PATH');
assert(~isempty(methods_path), 'DAG_ML_OCTAVE_METHODS_PATH must select the public Methods binding');
addpath(methods_path);
mex_path = getenv('DAG_ML_OCTAVE_MEX_PATH');
if ~isempty(mex_path), addpath(mex_path); end
config = decode(fileread(getenv('DAGML_OCTAVE_ROLE_CONFIG')));
controller = 'controller:methods.octave.regression';
plugin = 'dagml.methods.octave.regression';
version_id = '1.0.0';
assert(same_json(config.manifest, config.trusted_manifest), ...
    'Current signed controller differs from independently trusted manifest');
assert(strcmp(config.manifest.controller_id, controller) && ...
    strcmp(config.manifest.controller_version, version_id) && ...
    strcmp(config.manifest.operator_kind, 'model') && ...
    any(strcmp(ids(config.manifest.capabilities), 'consumes_oof_predictions')), ...
    'Invalid current Octave Methods controller manifest');
assert(islogical(config.allow_fit) && isscalar(config.allow_fit), 'allow_fit must be logical');
assert(config.allow_fit == ~isempty(config.targets), 'Replay must not receive target values');
config.target_names = ids(config.target_names);
assert(isstruct(config.sources) && ~isempty(fieldnames(config.sources)), 'Sources must be named');
source_keys = fieldnames(config.sources);
for i = 1:numel(source_keys)
    key = source_keys{i}; source = config.sources.(key);
    source.sample_ids = ids(source.sample_ids);
    source.feature_names = ids(source.feature_names);
    source.rows = matrix_rows(source.rows, numel(source.sample_ids), numel(source.feature_names));
    config.sources.(key) = source;
end
if config.allow_fit
    config.targets.sample_ids = ids(config.targets.sample_ids);
    config.targets.values = matrix_rows(config.targets.values, ...
        numel(config.targets.sample_ids), numel(config.target_names));
end
assert(isstruct(config.operators) && ~isempty(fieldnames(config.operators)), 'Operators must be named');
operator_keys = fieldnames(config.operators);
for i = 1:numel(operator_keys)
    steps = recipe(struct(), config.operators.(operator_keys{i}));
    probe = n4m.RolePipeline(native_steps(steps));
    probe.close();
end
models = containers.Map('KeyType', 'char', 'ValueType', 'any');
artifacts = containers.Map('KeyType', 'char', 'ValueType', 'any');
next_handle = 0;
closed = false;
cleanup = onCleanup(@close_all); %#ok<NASGU>
while true
    line = read_json_line();
    if ~ischar(line), break; end
    try
        frame = decode(line);
        assert(isstruct(frame) && frame.schema_version == 1, 'Unsupported process schema');
        switch char(frame.type)
            case 'init'
                assert(~closed && strcmp(frame.controller_id, controller), 'Foreign controller initialization');
                runtime = struct('octave', version(), 'role_pipeline', which('n4m.RolePipeline'), ...
                    'mex', which('n4m.n4m_role_pipeline_mex'), 'methods_version', n4m.version());
                assert(~isempty(runtime.mex), 'The public RolePipeline MEX is required');
                reply = struct('type', 'ack', 'schema_version', 1, 'status', 'initialized', 'runtime', runtime);
            case 'task'
                assert(~closed, 'Octave Methods controller is closed');
                task_text = json_member(line, 'task');
                seed = json_member(task_text, 'seed');
                assert(~isempty(regexp(seed, '^(0|[1-9][0-9]*)$', 'once')) && ...
                    (numel(seed) < 20 || (numel(seed) == 20 && ...
                    lex_le(seed, '18446744073709551615'))), 'Seed must be an exact u64 token');
                node_result = invoke(frame.task);
                reply = struct('type', 'result', 'schema_version', 1, 'result', node_result);
                encoded = jsonencode(reply);
                encoded = strrep(encoded, '"seed":"__DAGML_U64_SEED__"', ['"seed":' seed]);
                fprintf(1, '%s\n', encoded); fflush(1); continue;
            case 'portable_artifact'
                assert(~closed, 'Octave Methods controller is closed');
                reply = struct('type', 'portable_artifact', 'schema_version', 1, ...
                    'result', portable(frame.task));
            case 'close'
                close_all();
                reply = struct('type', 'ack', 'schema_version', 1, 'status', 'closed');
            otherwise
                error('Unsupported Octave Methods process frame');
        end
        fprintf(1, '%s\n', jsonencode(reply)); fflush(1);
        if strcmp(frame.type, 'close'), break; end
    catch problem
        fprintf(1, '%s\n', jsonencode(struct('type', 'error', 'schema_version', 1, ...
            'error', struct('code', 'octave_methods_refusal', 'message', problem.message))));
        fflush(1);
    end
end

    function audit(operation, node, sample_ids)
        if nargin < 2, node = NaN; end
        if nargin < 3, sample_ids = {}; end
        fid = fopen(config.audit_path, 'a'); assert(fid >= 0, 'Cannot write lifecycle audit');
        guard = onCleanup(@() fclose(fid)); %#ok<NASGU>
        fprintf(fid, '%s\n', jsonencode(struct('operation', operation, 'node_id', node, ...
            'sample_ids', {sample_ids})));
        assert(fflush(fid) == 0, 'Cannot flush lifecycle audit');
    end
    function dispose(entry)
        entry.model.close(); audit('dispose', entry.node_id);
    end
    function close_all()
        if closed, return; end
        keys = models.keys();
        for j = 1:numel(keys)
            entry = models(keys{j}); remove(models, keys{j});
            dispose(entry); audit('release', entry.node_id);
        end
        if artifacts.Count, remove(artifacts, artifacts.keys()); end
        closed = true;
    end
    function handle = keep(entry)
        assert(next_handle < 2^31 - 1, 'Octave Methods handle space exhausted');
        next_handle = next_handle + 1;
        models(sprintf('%d', next_handle)) = entry;
        handle = struct('handle', next_handle, 'kind', 'model', 'owner_controller', controller);
    end
    function block = join_blocks(blocks)
        assert(~isempty(blocks), 'No current feature blocks');
        sample_ids = blocks{1}.sample_ids; X = []; names = {};
        for j = 1:numel(blocks)
            b = blocks{j};
            assert(numel(b.sample_ids) == numel(sample_ids) && ...
                isempty(setxor(sample_ids, b.sample_ids)), 'Feature block sample coverage mismatch');
            [present, rows] = ismember(sample_ids, b.sample_ids); assert(all(present));
            X = [X b.X(rows, :)]; %#ok<AGROW>
            names = [names b.feature_names]; %#ok<AGROW>
        end
        block = struct('sample_ids', {sample_ids}, 'X', X, 'feature_names', {ids(names)});
    end
    function block = features(task, partition)
        views = task.data_views; keys = ordered_keys(views); blocks = {};
        for j = 1:numel(keys)
            key = keys{j}; view = views.(key);
            if ~strcmp(view.partition, partition), continue; end
            nonfit = any(strcmp(partition, {'fold_validation', 'predict'}));
            assert(islogical(view.include_augmented) && isscalar(view.include_augmented) && ...
                ~view.include_augmented && islogical(view.include_excluded) && ...
                isscalar(view.include_excluded) && (nonfit || ~view.include_excluded), ...
                'Augmented or excluded fitting views require a specialized controller');
            sample_ids = ids(view.sample_ids); sources = ids(view.source_ids);
            assert(numel(sources) == 1 && isfield(config.sources, sources{1}), ...
                'Native view requires one named current source');
            source = config.sources.(sources{1});
            [present, rows] = ismember(sample_ids, source.sample_ids);
            assert(all(present), 'Unknown source sample IDs');
            columns = 1:numel(source.feature_names);
            if isfield(view, 'columns') && ~isempty(view.columns)
                [present, columns] = ismember(ids(view.columns), source.feature_names);
                assert(all(present), 'Unknown source columns');
            end
            names = cellfun(@(name) [namespace(key) '/' name], source.feature_names(columns), ...
                'UniformOutput', false);
            blocks{end + 1} = struct('sample_ids', {sample_ids}, 'X', source.rows(rows, columns), ...
                'feature_names', {names}); %#ok<AGROW>
        end
        block = join_blocks(blocks);
    end
    function block = predictions(task, outer)
        inputs = task.prediction_inputs; keys = ordered_keys(inputs); blocks = {};
        suffixes = struct('FIT_CV', ':outer', 'REFIT', ':refit', 'PREDICT', ':predict');
        suffix = suffixes.(task.phase);
        for j = 1:numel(keys)
            key = keys{j}; input = inputs.(key);
            if outer, selected = endsWith(key, suffix);
            else, selected = isempty(regexp(key, ':(outer|refit|predict|test)$', 'once')); end
            if ~selected, continue; end
            sample_ids = ids(input.sample_ids);
            assert(strcmp(input.prediction_level, 'sample'), 'Prediction inputs require sample level');
            if strcmp(task.phase, 'FIT_CV') || (strcmp(task.phase, 'REFIT') && ~outer)
                assert(strcmp(input.partition, 'validation'), 'Meta training requires native validation OOF');
                if strcmp(task.phase, 'FIT_CV') && ~outer
                    assert(~any(strcmp(cells(input.fold_ids), task.fold_id)), ...
                        'Outer fold predictions cannot train its meta-model');
                end
            end
            if outer && ~strcmp(task.phase, 'FIT_CV')
                assert(any(strcmp(input.partition, {'test', 'final'})), 'Meta prediction requires off-fold rows');
            end
            width = input.prediction_width;
            assert(isscalar(width) && isfinite(width) && width >= 1 && width == fix(width));
            X = matrix_rows(input.values, numel(sample_ids), width);
            names = arrayfun(@(k) [namespace(key) '/' num2str(k)], 0:width-1, 'UniformOutput', false);
            blocks{end + 1} = struct('sample_ids', {sample_ids}, 'X', X, 'feature_names', {names}); %#ok<AGROW>
        end
        block = join_blocks(blocks);
    end
    function values = target_rows(sample_ids)
        assert(~isempty(config.targets), 'Inference must not access target values');
        [present, rows] = ismember(sample_ids, config.targets.sample_ids);
        assert(all(present), 'Unknown target sample IDs');
        values = config.targets.values(rows, :);
    end
    function out = result(task, block, model, refs, handles)
        if nargin < 4, refs = {}; handles = struct(); end
        values = model.predict(block.X, block.feature_names);
        values = matrix_rows(values, numel(block.sample_ids), numel(config.target_names));
        node = task.node_plan; fold = nullable(task.fold_id); variant = nullable(task.variant_id);
        prediction = struct('producer_node', node.node_id, 'partition', 'final', ...
            'fold_id', fold, 'sample_ids', {block.sample_ids}, 'values', {json_rows(values)}, ...
            'target_names', {config.target_names});
        if strcmp(task.phase, 'FIT_CV'), prediction.partition = 'validation'; end
        lineage = struct('record_id', ['lineage:methods-octave:' task.run_id ':' node.node_id ':' ...
            task.phase ':' fallback(task.variant_id, 'base') ':' fallback(task.fold_id, 'full')], ...
            'run_id', task.run_id, 'node_id', node.node_id, 'phase', task.phase, ...
            'controller_id', controller, 'controller_version', version_id, ...
            'variant_id', variant, 'fold_id', fold, 'branch_path', {cells(task.branch_path)}, ...
            'input_lineage', {{}}, 'artifact_refs', {refs}, ...
            'params_fingerprint', node.params_fingerprint, 'data_model_shape_fingerprint', NaN, ...
            'aggregation_policy_fingerprint', NaN, 'seed', '__DAGML_U64_SEED__', ...
            'unsafe_flags', {{}}, 'metrics', struct(), 'loss_attestations', {{}}, ...
            'early_stopping_records', {{}});
        out = struct('node_id', node.node_id, 'outputs', struct(), 'artifacts', {refs}, ...
            'artifact_handles', handles, 'predictions', {{prediction}}, 'lineage', lineage);
        if strcmp(task.phase, 'FIT_CV')
            units = cellfun(@(id) struct('level', 'sample', 'id', id), block.sample_ids, 'UniformOutput', false);
            out.regression_targets = {struct('level', 'sample', 'unit_ids', {units}, ...
                'values', {json_rows(target_rows(block.sample_ids))}, 'target_names', {config.target_names})};
        end
        audit(task.phase, node.node_id, block.sample_ids);
    end
    function check_artifact(ref, payload)
        digest = hash('sha256', char(payload));
        assert(strcmp(ref.controller_id, controller) && strcmp(ref.kind, 'methods_role_pipeline') && ...
            strcmp(ref.backend, 'raw') && strcmp(ref.plugin, plugin) && ...
            strcmp(ref.plugin_version, version_id) && strcmp(ref.content_fingerprint, digest) && ...
            strcmp(ref.uri, ['artifacts/' digest '.json']) && ref.size_bytes == numel(payload), ...
            'Methods RAW artifact owner, hash, size, URI or plugin mismatch');
        assert(numel(ids(ref.id)) == 1, 'Invalid artifact ID');
    end
    function out = portable(task)
        assert(task.schema_version == 1, 'Unsupported portable bridge schema');
        switch char(task.operation)
            case 'export_artifact_payload'
                assert(isKey(artifacts, task.artifact_id), 'Unknown Methods artifact'); audit('export');
                out = struct('operation', 'exported_artifact_payload', 'schema_version', 1, ...
                    'payload', {num2cell(double(artifacts(task.artifact_id)))});
            case 'release_hydrated_artifact_payload'
                h = task.handle; key = handle_key(h);
                assert(isKey(models, key), 'Unknown or released hydrated Methods handle');
                entry = models(key); remove(models, key); dispose(entry);
                audit('release', entry.node_id);
                out = struct('operation', 'released_hydrated_artifact_payload', 'schema_version', 1);
            case 'hydrate_artifact_payload'
                request = task.request;
                assert(strcmp(request.controller_id, controller) && isfield(config.operators, request.node_id), ...
                    'Portable Methods node or owner mismatch');
                payload = byte_array(task.payload); check_artifact(request.artifact, payload);
                text = native2unicode(payload, 'UTF-8');
                % Scan root keys before decoding, rejecting duplicate wrapper keys.
                json_member(text, 'schema'); saved = decode(text);
                fields = {'schema', 'node_id', 'params_fingerprint', 'target_names', 'steps', 'feature_names', 'states'};
                assert(isequal(sort(fieldnames(saved)), sort(fields(:))) && ...
                    strcmp(saved.schema, 'dagml.methods.regression.v1') && ...
                    strcmp(saved.node_id, request.node_id) && ...
                    strcmp(saved.params_fingerprint, request.params_fingerprint), 'Closed Methods RAW wrapper mismatch');
                assert(isequal(ids(saved.target_names), config.target_names), 'Saved target names mismatch');
                names = ids(saved.feature_names);
                steps = recipe(struct(), struct('type', 'N4mRolePipeline', 'steps', {cells(saved.steps)}));
                state_tokens = json_array_elements(json_member(text, 'states'));
                states = cellfun(@decode, state_tokens, 'UniformOutput', false);
                assert(~isempty(states) && numel(states) == numel(steps), 'Incomplete native state sequence');
                for k = 1:numel(states)
                    states{k} = byte_array(states{k});
                    assert(numel(states{k}) >= 4 && isequal(states{k}(1:4), uint8('N4ME')), 'Native N4ME state required');
                end
                model = n4m.RolePipeline(native_steps(steps), names);
                retained_key = '';
                try
                    model.importStates(states); regressor(model);
                    entry = struct('model', model, 'node_id', saved.node_id, ...
                        'params_fingerprint', saved.params_fingerprint, 'steps', {steps}, ...
                        'feature_names', {names}, 'artifact', request.artifact);
                    handle = keep(entry); retained_key = sprintf('%d', handle.handle);
                    audit('hydrate', saved.node_id);
                catch problem
                    if ~isempty(retained_key) && isKey(models, retained_key), remove(models, retained_key); end
                    model.close(); audit('dispose', saved.node_id); rethrow(problem);
                end
                out = struct('operation', 'hydrated_artifact_payload', 'schema_version', 1, 'handle', handle);
            otherwise
                error('Unsupported portable artifact operation');
        end
    end
    function key = handle_key(handle)
        assert(isstruct(handle) && strcmp(handle.owner_controller, controller) && ...
            strcmp(handle.kind, 'model') && isscalar(handle.handle) && ...
            isfinite(handle.handle) && handle.handle >= 1 && handle.handle == fix(handle.handle), ...
            'Unknown or foreign hydrated Methods handle');
        key = sprintf('%d', handle.handle);
    end
    function out = invoke(task)
        node = task.node_plan;
        assert(strcmp(node.kind, 'model') && strcmp(node.controller_id, controller) && ...
            strcmp(node.controller_version, version_id) && isfield(config.operators, node.node_id), ...
            'Current Methods controller or node mismatch');
        assert(any(strcmp(task.phase, {'FIT_CV', 'REFIT', 'PREDICT'})), 'Unsupported Methods phase');
        assert(empty_value(optional_member(task, 'data_view_receipts')) && empty_value(optional_member(task, 'required_loss_attestations')) && ...
            empty_value(optional_member(task, 'residual_targets')) && (empty_value(optional_member(task, 'fit_influence')) || ...
            (strcmp(task.fit_influence.mechanism, 'uniform_rows') && empty_value(optional_member(task.fit_influence, 'row_weights')) && ...
            empty_value(optional_member(task.fit_influence, 'target_row_weights')))), ...
            'Generated views, losses, residuals or nonuniform influence need a specialized controller');
        prediction_keys = fieldnames(task.prediction_inputs);
        assert(~strcmp(task.phase, 'FIT_CV') || ~any(endsWith(prediction_keys, ':test')), ...
            'Additional CV test streams need a specialized controller');
        meta = ~isempty(prediction_keys);
        steps = recipe(node, config.operators.(node.node_id));
        if strcmp(task.phase, 'PREDICT')
            keys = fieldnames(task.artifact_inputs);
            assert(numel(keys) == 1 && isfield(task.input_handles, keys{1}), 'PREDICT needs one attested model artifact');
            key = handle_key(task.input_handles.(keys{1}));
            assert(isKey(models, key), 'Unknown hydrated Methods handle'); entry = models(key);
            input = task.artifact_inputs.(keys{1});
            assert(strcmp(entry.node_id, node.node_id) && strcmp(input.node_id, node.node_id) && ...
                strcmp(input.controller_id, controller) && ...
                strcmp(entry.params_fingerprint, node.params_fingerprint) && ...
                strcmp(input.params_fingerprint, node.params_fingerprint) && ...
                same_json(input.artifact, entry.artifact) && ...
                same_json(entry.steps, steps), 'PREDICT owner, recipe or parameters mismatch');
            if meta, block = predictions(task, true); else, block = features(task, 'predict'); end
            assert(isequal(entry.feature_names, block.feature_names), 'Portable Methods feature order mismatch');
            out = result(task, block, entry.model); return;
        end
        assert(config.allow_fit, 'Fitting is disabled for this replay adapter');
        if meta, train = predictions(task, false);
        elseif strcmp(task.phase, 'FIT_CV'), train = features(task, 'fold_train');
        else, train = features(task, 'full_train'); end
        if strcmp(task.phase, 'FIT_CV')
            if meta, valid = predictions(task, true); else, valid = features(task, 'fold_validation'); end
        elseif meta && any(endsWith(prediction_keys, ':refit')), valid = predictions(task, true);
        else, valid = train; end
        assert(isequal(train.feature_names, valid.feature_names), 'Fit and validation feature order mismatch');
        assert(~strcmp(task.phase, 'FIT_CV') || isempty(intersect(train.sample_ids, valid.sample_ids)), ...
            'Fit and validation sample IDs overlap');
        model = n4m.RolePipeline(native_steps(steps), train.feature_names);
        retained = false;
        try
            model.fit(train.X, target_rows(train.sample_ids)); regressor(model);
            audit('fit', node.node_id, train.sample_ids);
            if strcmp(task.phase, 'FIT_CV')
                out = result(task, valid, model);
            else
                states = model.exportStates(false);
                assert(numel(states) == numel(steps), 'One N4ME state per declared recipe step required');
                states = cellfun(@(state) num2cell(double(state(:)')), states, 'UniformOutput', false);
                saved = struct('schema', 'dagml.methods.regression.v1', 'node_id', node.node_id, ...
                    'params_fingerprint', node.params_fingerprint, 'target_names', {config.target_names}, ...
                    'steps', {steps}, 'feature_names', {train.feature_names}, 'states', {states});
                payload = unicode2native(jsonencode(saved), 'UTF-8'); digest = hash('sha256', char(payload));
                artifact_id = ['artifact:methods:' task.run_id ':' node.node_id ':' fallback(task.variant_id, 'base') ':refit'];
                assert(~isKey(artifacts, artifact_id), 'Duplicate Methods REFIT artifact');
                ref = struct('id', artifact_id, 'kind', 'methods_role_pipeline', 'controller_id', controller, ...
                    'backend', 'raw', 'uri', ['artifacts/' digest '.json'], 'content_fingerprint', digest, ...
                    'size_bytes', numel(payload), 'plugin', plugin, 'plugin_version', version_id);
                handle = keep(struct('model', model, 'node_id', node.node_id, ...
                    'params_fingerprint', node.params_fingerprint, 'steps', {steps}, ...
                    'feature_names', {train.feature_names}, 'artifact', ref));
                retained = true; artifacts(artifact_id) = payload;
                handles = struct(); handles.(artifact_id) = handle;
                out = result(task, valid, model, {ref}, handles);
            end
        catch problem
            if retained
                key = sprintf('%d', handle.handle); remove(models, key); remove(artifacts, artifact_id);
            end
            dispose(struct('model', model, 'node_id', node.node_id)); rethrow(problem);
        end
        if ~retained, dispose(struct('model', model, 'node_id', node.node_id)); end
    end
end

function ok = same_json(a, b)
if isstruct(a) && isstruct(b) && isscalar(a) && isscalar(b)
    keys = sort(fieldnames(a));
    ok = isequal(keys, sort(fieldnames(b)));
    if ok
        for i = 1:numel(keys)
            if ~same_json(a.(keys{i}), b.(keys{i})), ok = false; return; end
        end
    end
elseif iscell(a) && iscell(b)
    ok = numel(a) == numel(b);
    if ok
        for i = 1:numel(a)
            if ~same_json(a{i}, b{i}), ok = false; return; end
        end
    end
else
    ok = isequaln(a, b);
end
end
function value = decode(text)
value = jsondecode(text, 'makeValidName', false);
end
function value = optional_member(object, name)
% Rust serde defaults omit empty receipts/losses/targets/influence fields.
if isfield(object, name), value = object.(name); else, value = []; end
end
function ok = empty_value(value)
if isstruct(value), ok = isempty(fieldnames(value)); else, ok = isempty(value); end
end
function value = nullable(value)
if isempty(value), value = NaN; end
end
function text = fallback(value, replacement)
if isempty(value), text = replacement; else, text = char(value); end
end
function value = cells(value)
if iscell(value), value = reshape(value, 1, []);
elseif isempty(value), value = {};
else, value = reshape(num2cell(value), 1, []); end
end
function names = ids(value)
if ischar(value), value = {value}; end
names = cells(value);
assert(~isempty(names) && all(cellfun(@(s) ischar(s) && ~isempty(strtrim(s)), names)) && ...
    numel(unique(names)) == numel(names), 'IDs/names must be unique nonempty strings');
end
function X = matrix_rows(value, rows, cols)
if iscell(value)
    value = cells(value); assert(numel(value) == rows, 'Numeric row count mismatch');
    X = zeros(rows, cols);
    for i = 1:rows
        row = value{i}; if iscell(row), row = cell2mat(row); end
        assert(isnumeric(row) && numel(row) == cols, 'Numeric column count mismatch');
        X(i, :) = reshape(row, 1, []);
    end
else
    assert(isnumeric(value) && numel(value) == rows * cols, 'Numeric matrix shape mismatch');
    if rows == 1 || cols == 1, X = reshape(value, rows, cols);
    else, assert(isequal(size(value), [rows cols]), 'Numeric matrix row/column order mismatch'); X = value; end
end
assert(all(isfinite(X(:))), 'Numeric matrices must be finite'); X = double(X);
end
function rows = json_rows(X)
rows = arrayfun(@(i) num2cell(X(i, :)), 1:size(X, 1), 'UniformOutput', false);
end
function bytes = byte_array(value)
if iscell(value), value = cell2mat(cells(value)); end
assert(isnumeric(value) && isvector(value) && ~isempty(value) && ...
    all(isfinite(value(:))) && all(value(:) == fix(value(:))) && ...
    all(value(:) >= 0 & value(:) <= 255) && numel(value) <= 134217728, 'Invalid bounded native byte array');
bytes = uint8(value(:)');
end
function prefix = namespace(key)
prefix = regexprep(key, ':(validation|outer|refit|predict|test)$', '');
end
function keys = ordered_keys(object)
assert(isstruct(object) && isscalar(object), 'Expected exact-key JSON object');
keys = fieldnames(object); tokens = cellfun(@(key) [namespace(key) char(1) key], keys, 'UniformOutput', false);
[~, order] = sort(tokens); keys = keys(order);
end
function steps = recipe(node, operator)
assert(isstruct(operator) && isfield(operator, 'type'), 'A current compiled Methods operator is required');
if strcmp(operator.type, 'N4mRolePipeline'), steps = cells(operator.steps);
else
    assert(startsWith(operator.type, 'n4m:') && numel(operator.type) > 4, 'Invalid native Methods operator');
    steps = {struct('class', operator.type, 'params', struct())};
end
assert(~isempty(steps) && numel(steps) <= 128, 'Methods recipes require 1 to 128 steps');
for i = 1:numel(steps)
    step = steps{i}; assert(isstruct(step) && isscalar(step));
    assert(all(ismember(fieldnames(step), {'class', 'methodId', 'params'})) && ...
        xor(isfield(step, 'class') && ~isempty(step.class), isfield(step, 'methodId') && ~isempty(step.methodId)), ...
        'Invalid closed native Methods recipe');
    if isfield(step, 'class') && ~isempty(step.class)
        assert(ischar(step.class) && startsWith(step.class, 'n4m:') && numel(step.class) > 4);
    else, assert(ischar(step.methodId) && ~isempty(step.methodId)); end
    if ~isfield(step, 'params'), step.params = struct(); end
    assert(isstruct(step.params) && isscalar(step.params), 'Methods parameters must be an object');
    steps{i} = step;
end
if isfield(node, 'params')
    assert(isstruct(node.params) && isscalar(node.params), 'Planned parameters must be an object');
    keys = fieldnames(node.params);
    for i = 1:numel(keys), steps{end}.params.(keys{i}) = node.params.(keys{i}); end
end
end
function translated = native_steps(steps)
translated = cell(size(steps));
for i = 1:numel(steps)
    step = steps{i};
    if isfield(step, 'class') && ~isempty(step.class), method_id = step.class(5:end);
    else, method_id = step.methodId; end
    translated{i} = struct('method_id', method_id, 'params', step.params);
end
end
function regressor(model)
info = model.stepsInfo(); assert(~isempty(info) && strcmp(info(end).role, 'regressor'), ...
    'Methods Octave controller accepts regressors only');
end
function line = read_json_line()
% Byte-wise input keeps a persistent pipe responsive before EOF, as the existing adapter.
line = '';
while true
    byte = fread(0, 1, 'char=>char');
    if isempty(byte), if isempty(line), line = -1; end; return; end
    if byte == char(10), return; end
    assert(numel(line) < 268435456, 'Process frame exceeds its bound');
    line(end + 1) = byte; %#ok<AGROW>
end
end
function token = json_member(text, wanted)
% Extract an exact top-level JSON token without converting a u64 to double.
i = 1; n = numel(text); keys = {}; token = '';
while i <= n && isspace(text(i)), i = i + 1; end
assert(i <= n && text(i) == '{', 'Expected JSON object'); i = i + 1;
while true
    while i <= n && (isspace(text(i)) || text(i) == ','), i = i + 1; end
    assert(i <= n, 'Incomplete JSON object');
    if text(i) == '}', break; end
    last = token_end(text, i); key = jsondecode(text(i:last));
    assert(ischar(key) && ~any(strcmp(keys, key)), 'Duplicate JSON object key'); keys{end + 1} = key;
    i = last + 1; while i <= n && isspace(text(i)), i = i + 1; end
    assert(i <= n && text(i) == ':', 'Expected JSON member colon'); i = i + 1;
    while i <= n && isspace(text(i)), i = i + 1; end
    last = token_end(text, i);
    if strcmp(key, wanted), token = strtrim(text(i:last)); end
    i = last + 1;
end
assert(~isempty(token), ['Missing JSON member ' wanted]);
end
function tokens = json_array_elements(text)
i = 1; n = numel(text); tokens = {};
while i <= n && isspace(text(i)), i = i + 1; end
assert(i <= n && text(i) == '[', 'Expected JSON array'); i = i + 1;
while true
    while i <= n && (isspace(text(i)) || text(i) == ','), i = i + 1; end
    assert(i <= n, 'Incomplete JSON array');
    if text(i) == ']', return; end
    last = token_end(text, i); tokens{end + 1} = text(i:last); i = last + 1;
end
end
function last = token_end(text, first)
in_string = false; escaped = false; depth = 0;
for i = first:numel(text)
    byte = text(i);
    if in_string
        if escaped, escaped = false;
        elseif byte == char(92), escaped = true;
        elseif byte == '"'
            in_string = false;
            if depth == 0, last = i; return; end
        end
    elseif byte == '"', in_string = true;
    elseif byte == '{' || byte == '[', depth = depth + 1;
    elseif byte == '}' || byte == ']'
        if depth == 0, last = i - 1; return; end
        depth = depth - 1; if depth == 0, last = i; return; end
    elseif depth == 0 && (byte == ',' || isspace(byte)), last = i - 1; return;
    end
end
last = numel(text);
end
function ok = lex_le(a, b)
index = find(a ~= b, 1); ok = isempty(index) || a(index) < b(index);
end
