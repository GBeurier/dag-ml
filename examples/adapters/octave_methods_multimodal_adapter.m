function octave_methods_multimodal_adapter()
% Persistent raw ND controller. Scaling/PCA/categories/fusion stay in Methods.
parts = sscanf(version(), '%d.%d');
assert(numel(parts) >= 2 && (parts(1) > 7 || (parts(1) == 7 && parts(2) >= 1)) && ...
    exist('jsondecode', 'builtin') == 5 && exist('jsonencode', 'builtin') == 5, ...
    'The product adapter requires Octave 7.1+ with native JSON support');
addpath(getenv('DAG_ML_OCTAVE_METHODS_PATH'));
mex_path = getenv('DAG_ML_OCTAVE_MEX_PATH');
if ~isempty(mex_path), addpath(mex_path); end
config_text = fileread(getenv('DAGML_METHODS_MULTIMODAL_CONFIG')); config = decode(config_text);
controller = 'controller:methods.octave.multimodal'; plugin = 'dagml.methods.octave.multimodal';
if isfield(config, 'controller_id'), controller = config.controller_id; end
hosts = {'python', 'wasm', 'r', 'octave'}; owner = find(cellfun(@(host) strcmp(controller, ...
    ['controller:methods.' host '.multimodal']), hosts));
assert(numel(owner) == 1 && (~config.allow_fit || strcmp(hosts{owner}, 'octave')), ...
    'Closed producer owner required; fitting requires Octave ownership');
plugin = ['dagml.methods.' hosts{owner} '.multimodal'];
source_order = {'nir', 'image', 'series', 'metadata'};
assert(same_json(config.manifest, config.trusted_manifest) && ...
    strcmp(config.manifest.controller_id, controller) && ...
    strcmp(config.manifest.controller_version, '1.0.0') && ...
    strcmp(config.manifest.operator_kind, 'model'), 'Untrusted current controller manifest');
assert(islogical(config.allow_fit) && isscalar(config.allow_fit) && ...
    config.allow_fit == ~isempty(config.targets), 'Replay cannot receive fitting targets');
config.target_names = ids(config.target_names); config.source_ids = ids(config.source_ids);
assert(numel(config.target_names) == 1 && numel(config.source_ids) == 4 && ...
    closed_keys(config.sources, source_order), 'One target and four raw sources required');
for i = 1:4
    name = source_order{i}; source = config.sources.(name); source.sample_ids = ids(source.sample_ids);
    shape = [numel(source.sample_ids) reshape(double(source.descriptor.input_shape), 1, [])];
    if strcmp(name, 'metadata')
        raw_metadata = json_member(json_member(config_text, 'sources'), 'metadata');
        row_tokens = json_array_elements(json_member(raw_metadata, 'rows'));
        rows = cellfun(@decode, row_tokens, 'UniformOutput', false);
        assert(numel(rows) == shape(1), 'Raw metadata row count mismatch');
        values = cell(shape(1), 2);
        for j = 1:shape(1)
            row = rows{j}; if ischar(row), row = cellstr(row); else, row = cells(row); end
            assert(numel(row) == 2 && ischar(row{2}) && ...
                numel(unicode2native(row{2}, 'UTF-8')) <= 1048576, 'Raw UTF-8 categorical cell required');
            numeric = row{1}; if ischar(numeric), numeric = str2double(numeric); end
            assert(number(numeric, -Inf, Inf, false), 'Finite raw numeric metadata required');
            values(j, :) = row;
        end
        source.values = values;
    else
        data = source.data; if iscell(data), data = cell2mat(cells(data)); end
        assert(isnumeric(data) && numel(data) == prod(shape) && numel(data) <= 16777216 && ...
            all(isfinite(data(:))) && isequal(reshape(double(source.shape), 1, []), shape), ...
            'Raw tensor shape or value budget mismatch');
        % Row-major transport to native column-major strides; no learned transform.
        if strcmp(source.descriptor.dtype, 'float32'), data = single(data);
        else
            assert(strcmp(source.descriptor.dtype, 'float64'), 'Unsupported declared numeric tensor dtype');
            data = double(data);
        end
        source.values = permute(reshape(data, fliplr(shape)), numel(shape):-1:1);
    end
    config.sources.(name) = source;
end
if config.allow_fit
    config.targets.sample_ids = ids(config.targets.sample_ids);
    config.targets.values = matrix_rows(config.targets.values, numel(config.targets.sample_ids), 1);
end
operators = fieldnames(config.operators); schemas = struct();
for i = 1:4, schemas.(source_order{i}) = config.sources.(source_order{i}).descriptor; end
for i = 1:numel(operators)
    op = config.operators.(operators{i}); params = struct();
    if isfield(config.node_params, operators{i}), params = config.node_params.(operators{i}); end
    recipe_for(op, params);
    assert(same_json(op.source_schemas, schemas), 'Current independently resolved raw schema mismatch');
end
models = containers.Map('KeyType', 'char', 'ValueType', 'any');
artifacts = containers.Map('KeyType', 'char', 'ValueType', 'any'); next_handle = 0; closed = false;
cleanup = onCleanup(@close_all); %#ok<NASGU>
while true
    line = read_json_line(); if ~ischar(line), break; end
    try
        frame = decode(line); assert(frame.schema_version == 1 && ~closed, 'Unsupported or closed process');
        switch char(frame.type)
            case 'init'
                assert(strcmp(frame.controller_id, controller), 'Foreign controller init');
                runtime = struct('execution_host', 'octave', 'signed_controller', controller, 'octave', version(), 'multimodal_pipeline', which('n4m.MultimodalPipeline'), ...
                    'mex', which('n4m.n4m_multimodal_pipeline_mex'), 'methods_version', n4m.version());
                assert(~isempty(runtime.mex), 'Public MultimodalPipeline MEX is required');
                reply = struct('type', 'ack', 'schema_version', 1, 'status', 'initialized', 'runtime', runtime);
            case 'task'
                seed = json_member(json_member(line, 'task'), 'seed');
                assert(strcmp(seed, 'null') || (~isempty(regexp(seed, '^(0|[1-9][0-9]*)$', 'once')) && ...
                    (numel(seed) < 20 || (numel(seed) == 20 && lex_le(seed, '18446744073709551615')))), ...
                    'Exact native optional u64 seed token required');
                encoded = jsonencode(struct('type', 'result', 'schema_version', 1, 'result', invoke(frame.task)));
                encoded = strrep(encoded, '"seed":"__DAGML_U64_SEED__"', ['"seed":' seed]);
                fprintf(1, '%s\n', encoded); fflush(1); continue;
            case 'portable_artifact'
                reply = struct('type', 'portable_artifact', 'schema_version', 1, 'result', portable(frame.task));
            case 'close'
                close_all(); reply = struct('type', 'ack', 'schema_version', 1, 'status', 'closed');
            otherwise, error('Unsupported multimodal process frame');
        end
        fprintf(1, '%s\n', jsonencode(reply)); fflush(1); if closed, break; end
    catch problem
        fprintf(1, '%s\n', jsonencode(struct('type', 'error', 'schema_version', 1, ...
            'error', struct('code', 'octave_methods_multimodal_refusal', 'message', problem.message)))); fflush(1);
    end
end
    function audit(operation, node, samples)
        if nargin < 2, node = NaN; end
        if nargin < 3, samples = {}; end
        fid = fopen(config.audit_path, 'a'); assert(fid >= 0, 'Cannot write lifecycle audit');
        guard = onCleanup(@() fclose(fid)); %#ok<NASGU>
        fprintf(fid, '%s\n', jsonencode(struct('operation', operation, 'node_id', node, 'sample_ids', {samples})));
    end
    function dispose(entry)
        entry.model.close(); audit('dispose', entry.saved.node_id);
    end
    function close_all()
        if closed, return; end
        names = models.keys(); failure = [];
        for k = 1:numel(names)
            entry = models(names{k}); remove(models, names{k});
            try, dispose(entry); catch problem, if isempty(failure), failure = problem; end; end
        end
        if artifacts.Count, remove(artifacts, artifacts.keys()); end
        closed = true; if ~isempty(failure), rethrow(failure); end
    end
    function handle = keep(entry)
        assert(next_handle < 2^31 - 1, 'Handle space exhausted'); next_handle = next_handle + 1;
        models(sprintf('%d', next_handle)) = entry;
        handle = struct('handle', next_handle, 'kind', 'model', 'owner_controller', controller);
    end
    function key = handle_key(handle)
        assert(strcmp(handle.kind, 'model') && strcmp(handle.owner_controller, controller) && ...
            number(handle.handle, 1, 2^31 - 1, true), 'Foreign model handle');
        key = sprintf('%d', handle.handle); assert(isKey(models, key), 'Unknown model handle');
    end
    function block = features(task, partition)
        names = fieldnames(task.data_views); selected = {};
        for k = 1:numel(names)
            v = task.data_views.(names{k}); if strcmp(v.partition, partition), selected{end + 1} = v; end %#ok<AGROW>
        end
        assert(numel(selected) == 1, 'One native raw multimodal view required'); view = selected{1};
        samples = ids(view.sample_ids);
        assert(isequal(ids(view.source_ids), config.source_ids) && ~view.include_augmented && ...
            (any(strcmp(partition, {'fold_validation', 'predict'})) || ~view.include_excluded) && isempty(view.columns), 'Native raw source view mismatch');
        blocks = struct();
        chosen = ids(config.operators.(task.node_plan.node_id).recipe.source_order);
        for k = 1:4
            name = source_order{k}; s = config.sources.(name); [present, rows] = ismember(samples, s.sample_ids);
            assert(all(present), 'Unknown raw sample ID');
            if ~any(strcmp(name, chosen)), continue; end
            indices = repmat({':'}, 1, numel(s.descriptor.input_shape) + 1); indices{1} = rows;
            blocks.(name) = s.values(indices{:});
        end
        blocks = selected_schemas(config.operators.(task.node_plan.node_id).recipe, blocks);
        block = struct('sample_ids', {samples}, 'blocks', blocks);
    end
    function y = targets(samples)
        assert(~isempty(config.targets), 'Replay cannot read training targets');
        [present, rows] = ismember(samples, config.targets.sample_ids); assert(all(present), 'Unknown target sample ID');
        y = config.targets.values(rows, :);
    end
    function out = result(task, block, model, refs)
        if nargin < 4, refs = {}; end
        node = task.node_plan; values = matrix_rows(model.predict(block.blocks, ...
            selected_schemas(config.operators.(node.node_id).recipe, config.operators.(node.node_id).source_schemas)), numel(block.sample_ids), 1);
        prediction = struct('producer_node', node.node_id, 'partition', 'final', 'fold_id', nullable(task.fold_id), ...
            'sample_ids', {block.sample_ids}, 'values', {json_rows(values)}, 'target_names', {config.target_names});
        if strcmp(task.phase, 'FIT_CV'), prediction.partition = 'validation'; end
        lineage = struct('record_id', bounded_identifier({'lineage:methods-multimodal', task.run_id, node.node_id, task.phase, ...
            fallback(task.variant_id, 'base'), fallback(task.fold_id, 'full')}), 'run_id', task.run_id, ...
            'node_id', node.node_id, 'phase', task.phase, 'controller_id', controller, 'controller_version', '1.0.0', ...
            'variant_id', nullable(task.variant_id), 'fold_id', nullable(task.fold_id), 'branch_path', {cells(task.branch_path)}, ...
            'input_lineage', {{}}, 'artifact_refs', {refs}, 'params_fingerprint', node.params_fingerprint, ...
            'data_model_shape_fingerprint', NaN, 'aggregation_policy_fingerprint', NaN, 'seed', '__DAGML_U64_SEED__', ...
            'unsafe_flags', {{}}, 'metrics', struct(), 'loss_attestations', {{}}, 'early_stopping_records', {{}});
        out = struct('node_id', node.node_id, 'outputs', struct(), 'artifacts', {refs}, ...
            'artifact_handles', struct(), 'predictions', {{prediction}}, 'lineage', lineage);
        if ~isempty(config.targets)
            units = cellfun(@(id) struct('level', 'sample', 'id', id), block.sample_ids, 'UniformOutput', false);
            out.regression_targets = {struct('level', 'sample', 'unit_ids', {units}, ...
                'values', {json_rows(targets(block.sample_ids))}, 'target_names', {config.target_names})};
        end
        views = struct2cell(task.data_views);
        if any(strcmp(task.phase, {'FIT_CV', 'REFIT'})) && any(cellfun(@(v) strcmp(v.partition, 'predict'), views))
            test = features(task, 'predict'); values = matrix_rows(model.predict(test.blocks, ...
                selected_schemas(config.operators.(node.node_id).recipe, config.operators.(node.node_id).source_schemas)), numel(test.sample_ids), 1);
            out.predictions{end + 1} = struct('producer_node', node.node_id, 'partition', 'test', ...
                'fold_id', nullable(task.fold_id), 'sample_ids', {test.sample_ids}, 'values', {json_rows(values)}, ...
                'target_names', {config.target_names});
            if ~isempty(config.targets)
                units = cellfun(@(id) struct('level', 'sample', 'id', id), test.sample_ids, 'UniformOutput', false);
                out.regression_targets{end + 1} = struct('level', 'sample', 'unit_ids', {units}, ...
                    'values', {json_rows(targets(test.sample_ids))}, 'target_names', {config.target_names});
            end
        end
        audit(task.phase, node.node_id, block.sample_ids);
    end
    function out = portable(task)
        assert(task.schema_version == 1, 'Unsupported artifact bridge');
        switch char(task.operation)
            case 'export_artifact_payload'
                assert(isKey(artifacts, task.artifact_id), 'Unknown artifact'); audit('export');
                out = struct('operation', 'exported_artifact_payload', 'schema_version', 1, ...
                    'payload', {num2cell(double(artifacts(task.artifact_id)))});
            case 'release_hydrated_artifact_payload'
                key = handle_key(task.handle); entry = models(key); remove(models, key); dispose(entry); audit('release');
                out = struct('operation', 'released_hydrated_artifact_payload', 'schema_version', 1);
            case 'hydrate_artifact_payload'
                payload = byte_array(task.payload); request = task.request; ref = request.artifact;
                digest = hash('sha256', char(payload));
                assert(strcmp(request.controller_id, controller) && strcmp(ref.controller_id, controller) && ...
                    strcmp(ref.kind, 'methods_multimodal_pipeline') && strcmp(ref.backend, 'raw') && ...
                    strcmp(ref.plugin, plugin) && strcmp(ref.plugin_version, '1.0.0') && ...
                    (~isfield(ref, 'native_predictor_descriptor') || isempty(ref.native_predictor_descriptor)) && ...
                    (~isfield(ref, 'native_estimator_descriptor') || isempty(ref.native_estimator_descriptor)) && ...
                    strcmp(ref.content_fingerprint, digest) && strcmp(ref.uri, ['artifacts/' digest '.json']) && ...
                    ref.size_bytes == numel(payload), 'RAW owner/plugin/SHA/URI/size mismatch');
                saved = decode(native2unicode(payload, 'UTF-8')); state = validate_wrapper(saved);
                assert(isfield(config.operators, saved.node_id), 'Foreign saved node'); op = config.operators.(saved.node_id);
                params = struct(); if isfield(config.node_params, saved.node_id), params = config.node_params.(saved.node_id); end
                assert(strcmp(saved.node_id, request.node_id) && strcmp(saved.params_fingerprint, request.params_fingerprint) && ...
                    isequal(ids(saved.target_names), config.target_names) && same_json(saved.source_schemas, op.source_schemas) && ...
                    same_json(saved.recipe, recipe_for(op, params)), 'Selected recipe/schema/node mismatch before hydration');
                model = n4m.MultimodalPipeline.fromState(state, native_recipe(saved.recipe), native_schemas(selected_schemas(saved.recipe, saved.source_schemas)));
                try, handle = keep(struct('model', model, 'saved', saved, 'artifact', ref));
                catch problem, model.close(); rethrow(problem); end
                audit('hydrate'); out = struct('operation', 'hydrated_artifact_payload', 'schema_version', 1, 'handle', handle);
            otherwise, error('Unsupported complete predictor artifact operation');
        end
    end
    function out = invoke(task)
        node = task.node_plan;
        assert(strcmp(node.kind, 'model') && strcmp(node.controller_id, controller) && ...
            strcmp(node.controller_version, '1.0.0') && isfield(config.operators, node.node_id) && ...
            any(strcmp(task.phase, {'FIT_CV', 'REFIT', 'PREDICT'})), 'Foreign native node or phase');
        assert(empty_value(task.prediction_inputs) && empty_value(task.data_view_receipts) && ...
            empty_value(task.required_loss_attestations) && empty_value(task.residual_targets) && ...
            (empty_value(task.fit_influence) || (strcmp(task.fit_influence.mechanism, 'uniform_rows') && ...
            isempty(task.fit_influence.row_weights))), 'Generated/OOF/loss/residual/nonuniform inputs unsupported');
        op = config.operators.(node.node_id); params = struct();
        if isfield(node, 'params'), params = node.params; end
        recipe = recipe_for(op, params);
        if strcmp(task.phase, 'PREDICT')
            names = fieldnames(task.artifact_inputs); assert(numel(names) == 1, 'One complete predictor required');
            input = task.artifact_inputs.(names{1}); entry = models(handle_key(task.input_handles.(names{1})));
            assert(strcmp(input.node_id, node.node_id) && strcmp(input.controller_id, controller) && ...
                strcmp(input.params_fingerprint, node.params_fingerprint) && same_json(input.artifact, entry.artifact) && ...
                strcmp(entry.saved.node_id, node.node_id) && strcmp(entry.saved.params_fingerprint, node.params_fingerprint) && ...
                same_json(entry.saved.recipe, recipe) && same_json(entry.saved.source_schemas, op.source_schemas), ...
                'PREDICT predictor binding mismatch'); out = result(task, features(task, 'predict'), entry.model); return;
        end
        assert(config.allow_fit, 'Fitting disabled for replay');
        if strcmp(task.phase, 'FIT_CV'), train = features(task, 'fold_train'); valid = features(task, 'fold_validation');
        else, train = features(task, 'full_train'); valid = train; end
        assert(~strcmp(task.phase, 'FIT_CV') || isempty(intersect(train.sample_ids, valid.sample_ids)), 'Training validation overlap');
        model = n4m.MultimodalPipeline(native_recipe(recipe), native_schemas(selected_schemas(recipe, op.source_schemas))); retained = false;
        try
            model.fit(train.blocks, targets(train.sample_ids)); audit('fit', node.node_id, train.sample_ids);
            if strcmp(task.phase, 'FIT_CV'), out = result(task, valid, model);
            else
                saved = struct('schema', 'dagml.methods.multimodal.v1', 'node_id', node.node_id, ...
                    'params_fingerprint', node.params_fingerprint, 'target_names', {config.target_names}, ...
                    'recipe', recipe, 'source_schemas', op.source_schemas, 'state', {num2cell(double(model.exportState()))});
                validate_wrapper(saved); wire_saved = saved;
                wire_saved.recipe = wire_recipe(saved.recipe); wire_saved.source_schemas = native_schemas(saved.source_schemas);
                payload = unicode2native(jsonencode(wire_saved, 'ConvertInfAndNaN', true), 'UTF-8');
                assert(numel(payload) <= 134217728, 'Complete payload budget exceeded'); digest = hash('sha256', char(payload));
                id = bounded_identifier({'artifact:methods.multimodal', task.run_id, node.node_id, fallback(task.variant_id, 'base'), 'refit'});
                assert(~isKey(artifacts, id), 'Duplicate REFIT artifact');
                ref = struct('id', id, 'kind', 'methods_multimodal_pipeline', 'controller_id', controller, ...
                    'backend', 'raw', 'uri', ['artifacts/' digest '.json'], 'content_fingerprint', digest, ...
                    'size_bytes', numel(payload), 'plugin', plugin, 'plugin_version', '1.0.0');
                out = result(task, valid, model, {ref}); handle = keep(struct('model', model, 'saved', saved, 'artifact', ref));
                artifacts(id) = payload; out.artifact_handles.(id) = handle; retained = true;
            end
        catch problem
            model.close(); audit('dispose', node.node_id); rethrow(problem);
        end
        if ~retained, model.close(); audit('dispose', node.node_id); end
    end
end

function out = wire_recipe(recipe)
% JSON null differs from the empty no-drop argument required by the MEX.
out = native_recipe(recipe);
if isfield(out.encoders, 'metadata'), out.encoders.metadata.drop = NaN; end
end
function out = native_recipe(recipe)
% Restore wire arrays collapsed by jsondecode; values/semantics are unchanged.
out = recipe;
if isfield(recipe.encoders, 'metadata')
    out.encoders.metadata.numeric_columns = num2cell(double(recipe.encoders.metadata.numeric_columns(:)'));
    out.encoders.metadata.categorical_columns = num2cell(double(recipe.encoders.metadata.categorical_columns(:)'));
    out.encoders.metadata.drop = [];
end
end
function out = selected_schemas(recipe, schemas)
out = struct(); names = ids(recipe.source_order);
for i = 1:numel(names), out.(names{i}) = schemas.(names{i}); end
end
function out = native_schemas(schemas)
out = schemas; names = fieldnames(schemas);
for i = 1:numel(names), out.(names{i}).input_shape = num2cell(double(schemas.(names{i}).input_shape(:)')); end
end
function ok = closed_keys(value, names)
ok = isstruct(value) && isscalar(value) && isequal(sort(fieldnames(value)), sort(names(:)));
end
function ok = number(value, lower, upper, integer)
ok = isnumeric(value) && isscalar(value) && isfinite(value) && value >= lower && value <= upper && (~integer || value == fix(value));
end
function recipe = recipe_for(operator, params)
assert(closed_keys(operator, {'type', 'recipe', 'source_schemas'}) && strcmp(operator.type, 'N4mMultimodalPipeline') && ...
    isstruct(params) && all(ismember(fieldnames(params), {'model__alpha', 'source_weights__image', 'transformers__image__n_components', 'recipe', 'source_schemas'})), ...
    'Explicit native operator/effective parameters required');
assert(isfield(params, 'recipe') == isfield(params, 'source_schemas'), 'Both immutable declarations required');
assert(~isfield(params, 'recipe') || all(ismember(fieldnames(params), {'recipe', 'source_schemas', 'model__alpha'})), 'Structural multimodal tuning permits alpha only');
for name = {'recipe', 'source_schemas'}
    if isfield(params, name{1}), assert(same_json(params.(name{1}), operator.(name{1})), 'Structural declaration differs from signed operator'); end
end
recipe = operator.recipe;
if isfield(params, 'model__alpha'), recipe.model.params.alpha = params.model__alpha; end
if isfield(params, 'source_weights__image'), assert(isfield(recipe.source_weights, 'image'), 'Inactive image parameter'); recipe.source_weights.image = params.source_weights__image; end
if isfield(params, 'transformers__image__n_components'), assert(isfield(recipe.encoders, 'image'), 'Inactive image parameter'); recipe.encoders.image.n_components = params.transformers__image__n_components; end
validate_recipe(recipe, operator.source_schemas);
end
function validate_recipe(recipe, schemas)
names = {'nir', 'image', 'series', 'metadata'}; representations = {'signal_1d', 'rgb_image', 'series_mv', 'tabular_mixed'};
selected = ids(recipe.source_order);
assert(closed_keys(recipe, {'schema_version', 'fusion', 'source_order', 'encoders', 'source_weights', 'model'}) && ...
    number(recipe.schema_version, 1, 1, true) && strcmp(recipe.fusion, 'early') && all(ismember(selected, names)) && ...
    closed_keys(schemas, names) && closed_keys(recipe.encoders, selected) && closed_keys(recipe.source_weights, selected), ...
    'Closed selected recipe and complete raw schemas required');
for i = 1:4
    s = schemas.(names{i}); assert(closed_keys(s, {'representation_id', 'input_shape', 'dtype', 'identity'}));
    shape = double(s.input_shape(:));
    assert(strcmp(s.representation_id, representations{i}) && isnumeric(s.input_shape) && ~isempty(shape) && ...
        numel(shape) <= 7 && all(isfinite(shape) & shape > 0 & shape == fix(shape)) && prod(shape) <= 1048576 && ...
        (i ~= 4 || isequal(shape, 2)), 'Fixed declared raw shape required');
    assert(ischar(s.dtype) && numel(unicode2native(s.dtype, 'UTF-8')) >= 1 && numel(unicode2native(s.dtype, 'UTF-8')) <= 128 && ...
        ischar(s.identity) && numel(unicode2native(s.identity, 'UTF-8')) >= 1 && numel(unicode2native(s.identity, 'UTF-8')) <= 1048576, ...
        'Source identity budget exceeded'); decode(s.identity);
end
for i = 1:numel(selected), assert(number(recipe.source_weights.(selected{i}), 0, Inf, false), 'Finite nonnegative source weight required'); end
if isfield(recipe.encoders, 'nir')
    assert(same_json(recipe.encoders.nir, struct('kind', 'standard_scaler', 'with_mean', true, 'with_std', true)), 'Closed population scaler required');
end
if isfield(recipe.encoders, 'metadata')
m = recipe.encoders.metadata;
assert(closed_keys(m, {'kind', 'numeric_columns', 'categorical_columns', 'with_mean', 'with_std', 'handle_unknown', 'sparse_output', 'drop'}) && ...
    strcmp(m.kind, 'column_transformer') && isequal(m.numeric_columns, 0) && isequal(m.categorical_columns, 1) && ...
    isequal(m.with_mean, true) && isequal(m.with_std, true) && strcmp(m.handle_unknown, 'ignore') && ...
    isequal(m.sparse_output, false) && empty_value(m.drop), 'Closed native mixed encoder required');
end
for name = {'image', 'series'}
    if ~isfield(recipe.encoders, name{1}), continue; end
    e = recipe.encoders.(name{1}); assert(closed_keys(e, {'kind', 'n_components', 'whiten', 'random_state'}) && ...
        strcmp(e.kind, 'tensor_pca') && number(e.n_components, 1, min(2^31 - 1, prod(schemas.(name{1}).input_shape)), true) && ...
        isequal(e.whiten, false) && number(e.random_state, 0, 2^32 - 1, true), 'Declared unwhitened PCA required');
end
m = recipe.model; assert(closed_keys(m, {'method_id', 'params'}) && strcmp(m.method_id, 'models.regularized.ridge') && ...
    closed_keys(m.params, {'alpha', 'center_x', 'center_y', 'scale_x'}) && number(m.params.alpha, 0, Inf, false) && ...
    isequal(m.params.center_x, true) && isequal(m.params.center_y, true) && isequal(m.params.scale_x, false), 'Closed native Ridge recipe required');
end
function state = validate_wrapper(saved)
assert(closed_keys(saved, {'schema', 'node_id', 'params_fingerprint', 'target_names', 'recipe', 'source_schemas', 'state'}) && ...
    strcmp(saved.schema, 'dagml.methods.multimodal.v1') && numel(ids(saved.target_names)) == 1, 'Closed predictor wrapper required');
state = byte_array(saved.state); assert(numel(state) >= 28 && numel(state) <= 67108864 && ...
    isequal(double(state(1:12)), [78 52 77 70 1 0 0 0 2 0 0 0]), 'Unsupported N4MF header');
validate_recipe(saved.recipe, saved.source_schemas);
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
strict_json(text, 0);
value = jsondecode(text, 'makeValidName', false);
end
function ok = empty_value(value)
if isstruct(value), ok = isempty(fieldnames(value));
else, ok = isempty(value) || (isnumeric(value) && isscalar(value) && isnan(value)); end
end
function identifier = bounded_identifier(parts)
identifier = strjoin(parts, ':');
if numel(unicode2native(identifier, 'UTF-8')) <= 128, return; end
payload = unicode2native(jsonencode(parts), 'UTF-8');
identifier = [parts{1} ':' hash('sha256', char(payload))];
end
function value = nullable(value)
if isempty(value), value = NaN; end
end
function text = fallback(value, replacement)
if empty_value(value), text = replacement; else, text = char(value); end
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
function strict_json(text, depth)
% Validate every nested key before jsondecode can overwrite duplicates.
assert(depth <= 128 && numel(text) <= 134217728, 'JSON nesting/transport budget exceeded');
text = strtrim(text); assert(~isempty(text), 'Empty JSON');
if text(1) == '{'
    i = 2; keys = {};
    while true
        while i <= numel(text) && isspace(text(i)), i = i + 1; end
        assert(i <= numel(text), 'Incomplete JSON object'); if text(i) == '}', break; end
        assert(text(i) == '"', 'Expected JSON key'); last = token_end(text, i); key = jsondecode(text(i:last));
        assert(~any(strcmp(keys, key)), 'Duplicate JSON key'); keys{end + 1} = key; %#ok<AGROW>
        i = last + 1; while i <= numel(text) && isspace(text(i)), i = i + 1; end
        assert(text(i) == ':', 'Expected JSON colon'); i = i + 1;
        while i <= numel(text) && isspace(text(i)), i = i + 1; end
        last = token_end(text, i); strict_json(text(i:last), depth + 1); i = last + 1;
        while i <= numel(text) && isspace(text(i)), i = i + 1; end
        if text(i) == '}', break; end
        assert(text(i) == ',', 'Expected comma'); i = i + 1;
    end
elseif text(1) == '['
    values = json_array_elements(text); for i = 1:numel(values), strict_json(values{i}, depth + 1); end
elseif text(1) ~= '"' && ~any(strcmp(text, {'null', 'true', 'false'}))
    assert(~isempty(regexp(text, '^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$', 'once')) && ...
        isfinite(str2double(text)), 'Invalid or nonfinite JSON token');
end
end
