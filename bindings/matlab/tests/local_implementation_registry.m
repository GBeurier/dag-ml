function local_implementation_registry()
% Exercise MATLAB-owned loss and metric functions against native NodeTasks.

bindingRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(bindingRoot);
fixturePath = fullfile(bindingRoot, 'fixtures', ...
    'matlab_local_implementations.v1.json');
fixture = jsondecode(fileread(fixturePath));

global DAGML_TEST_LOSS_CALLS;
DAGML_TEST_LOSS_CALLS = 0;
registry = dagml.LocalImplementationRegistry();
registry.registerLoss(fixture.loss_reference, @asymmetricLoss);
registry.registerMetric(fixture.metric_reference, @biasMetric);
assert(registry.count() == 2);
assert(numel(registry.descriptors()) == 2);

phases = {'FIT_CV', 'REFIT'};
for index = 1:numel(phases)
    phase = phases{index};
    [value, attestation] = registry.invokeTrainingLoss( ...
        fixture.task_json.(phase), 1, [2, 4], [5, 3]);
    assert(abs(value - 5.5) < eps);
    assert(strcmp(attestation.phase, phase));
    assert(strcmp(attestation.descriptor_fingerprint, ...
        fixture.loss_reference.implementation.descriptor_fingerprint));
    assert(isequal(attestation, ...
        fixture.tasks.(phase).required_loss_attestations));
end
assert(DAGML_TEST_LOSS_CALLS == 2);

metricValue = registry.invokeMetric( ...
    fixture.metric_reference, [2, 4], [5, 3]);
assert(abs(metricValue - 1) < eps);

assertThrows(@() registry.invokeTrainingLoss( ...
    fixture.invalid_task_json.predict, 1, 2, 5), 'FIT_CV or REFIT');
assert(DAGML_TEST_LOSS_CALLS == 2);

phaseControllers = containers.Map();
phaseControllers('controller:matlab-local') = ...
    @(controllerId, taskJson) error('dagml:test:Unexpected', ...
    'unexpected callback'); %#ok<NASGU>
assertThrows(@() dagml.executeExecutionPlanPhase( ...
    '{}', {}, 'run:matlab-phase', 1, 'FIT_CV', phaseControllers, ''), ...
    'native library');

assertThrows(@() registry.registerLoss( ...
    fixture.foreign_loss_reference, @asymmetricLoss), 'binding:matlab');

portableBuiltin = fixture.loss_reference;
portableBuiltin.implementation.portability = 'portable_builtin';
portableRegistry = dagml.LocalImplementationRegistry();
assertThrows(@() portableRegistry.registerLoss( ...
    portableBuiltin, @asymmetricLoss), 'portable_builtin');
assertThrows(@() registry.registerLoss( ...
    fixture.loss_reference, @asymmetricLoss), 'Duplicate');
assertThrows(@() registry.registerLoss( ...
    fixture.loss_reference, 42), 'function handles');
assertThrows(@() registry.resolveMetric( ...
    fixture.loss_reference), 'metric implementation descriptor');

drifted = fixture.loss_reference;
drifted.implementation.implementation_version = '2.0.0';
assertThrows(@() registry.resolveLoss(drifted), 'does not match');

assertThrows(@() registry.invokeTrainingLoss( ...
    fixture.invalid_task_json.tampered_attestation, 1, 2, 5), ...
    'requirements that do not match');
assert(DAGML_TEST_LOSS_CALLS == 2);

assertThrows(@() registry.invokeTrainingLoss( ...
    fixture.invalid_task_json.wrong_attestation_schema, 1, 2, 5), ...
    'requirements that do not match');
assert(DAGML_TEST_LOSS_CALLS == 2);

assertThrows(@() registry.invokeTrainingLoss( ...
    fixture.invalid_task_json.missing_attestation, 1, 2, 5), ...
    'requirements that do not match');
assertThrows(@() registry.invokeTrainingLoss( ...
    fixture.task_json.FIT_CV, 2, 2, 5), 'outside the active');
assert(DAGML_TEST_LOSS_CALLS == 2);

withoutNative = dagml.LocalImplementationRegistry('');
withoutNative.registerLoss(fixture.loss_reference, @asymmetricLoss);
assertThrows(@() withoutNative.invokeTrainingLoss( ...
    fixture.task_json.FIT_CV, 1, 2, 5), 'DAGML_NATIVE_LIBRARY');
assert(DAGML_TEST_LOSS_CALLS == 2);

invalidNative = dagml.LocalImplementationRegistry('/dagml/does/not/exist');
invalidNative.registerLoss(fixture.loss_reference, @asymmetricLoss);
assertThrows(@() invalidNative.invokeTrainingLoss( ...
    fixture.task_json.FIT_CV, 1, 2, 5), ...
    'Failed to load DAG-ML native library');
assert(DAGML_TEST_LOSS_CALLS == 2);

assertThrows(@() registry.invokeTrainingLoss( ...
    fixture.tasks.FIT_CV, 1, 2, 5), 'NodeTask JSON');
assert(DAGML_TEST_LOSS_CALLS == 2);

failing = dagml.LocalImplementationRegistry();
failing.registerLoss(fixture.loss_reference, @failingLoss);
assertThrows(@() failing.invokeTrainingLoss( ...
    fixture.task_json.FIT_CV, 1, 2, 5), 'local failure');

assertThrows(@() registry.toJSON(), 'cannot be serialized');
assertThrows(@() registry.saveobj(), 'cannot be serialized');

removed = registry.unregisterLoss(fixture.loss_reference);
assert(isequal(removed, @asymmetricLoss));
assert(registry.count() == 1);
removedMetric = registry.unregisterMetric(fixture.metric_reference);
assert(isequal(removedMetric, @biasMetric));
assert(registry.count() == 0);
registry.registerMetric(fixture.metric_reference, @biasMetric);
registry.clear();
assert(registry.count() == 0);
clear global DAGML_TEST_LOSS_CALLS;
end

function value = asymmetricLoss(target, prediction)
global DAGML_TEST_LOSS_CALLS;
DAGML_TEST_LOSS_CALLS = DAGML_TEST_LOSS_CALLS + 1;
difference = prediction - target;
weights = ones(size(difference));
weights(difference < 0) = 2;
value = mean(weights .* difference .^ 2);
end

function value = biasMetric(target, prediction)
value = mean(prediction - target);
end

function value = failingLoss(varargin) %#ok<INUSD,STOUT>
error('dagml:test:LocalFailure', 'local failure');
end

function assertThrows(callback, messagePart)
didThrow = false;
try
    callback();
catch exception
    didThrow = true;
    assert(~isempty(strfind(exception.message, messagePart))); %#ok<STREMP>
end
assert(didThrow);
end
