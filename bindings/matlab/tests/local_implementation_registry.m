function local_implementation_registry()
% Exercise MATLAB-owned loss and metric functions against native NodeTasks.

bindingRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(bindingRoot);
fixturePath = fullfile(bindingRoot, 'fixtures', ...
    'matlab_local_implementations.v1.json');
fixture = jsondecode(fileread(fixturePath));

calls = 0;
registry = dagml.LocalImplementationRegistry();
registry.registerLoss(fixture.loss_reference, @asymmetricLoss);
registry.registerMetric(fixture.metric_reference, @biasMetric);
assert(registry.count() == 2);
assert(numel(registry.descriptors()) == 2);

phases = {'FIT_CV', 'REFIT'};
for index = 1:numel(phases)
    phase = phases{index};
    [value, attestation] = registry.invokeTrainingLoss( ...
        fixture.tasks.(phase), 1, [2, 4], [5, 3]);
    assert(abs(value - 5.5) < eps);
    assert(strcmp(attestation.phase, phase));
    assert(strcmp(attestation.descriptor_fingerprint, ...
        fixture.loss_reference.implementation.descriptor_fingerprint));
end
assert(calls == 2);

metricValue = registry.invokeMetric( ...
    fixture.metric_reference, [2, 4], [5, 3]);
assert(abs(metricValue - 1) < eps);

predictTask = fixture.tasks.FIT_CV;
predictTask.phase = 'PREDICT';
assertThrows(@() registry.invokeTrainingLoss( ...
    predictTask, 1, 2, 5), 'FIT_CV or REFIT');
assert(calls == 2);

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

tamperedTask = fixture.tasks.FIT_CV;
tamperedTask.required_loss_attestations.implementation_fingerprint = 'tampered';
assertThrows(@() registry.invokeTrainingLoss( ...
    tamperedTask, 1, 2, 5), 'implementation_fingerprint');
assert(calls == 2);

wrongSchemaTask = fixture.tasks.FIT_CV;
wrongSchemaTask.required_loss_attestations.schema_version = 2;
assertThrows(@() registry.invokeTrainingLoss( ...
    wrongSchemaTask, 1, 2, 5), 'schema_version');
assert(calls == 2);

missingRequirementTask = fixture.tasks.FIT_CV;
missingRequirementTask.required_loss_attestations = struct([]);
assertThrows(@() registry.invokeTrainingLoss( ...
    missingRequirementTask, 1, 2, 5), 'count does not match');
assertThrows(@() registry.invokeTrainingLoss( ...
    fixture.tasks.FIT_CV, 2, 2, 5), 'outside the active');
assert(calls == 2);

failing = dagml.LocalImplementationRegistry();
failing.registerLoss(fixture.loss_reference, @failingLoss);
assertThrows(@() failing.invokeTrainingLoss( ...
    fixture.tasks.FIT_CV, 1, 2, 5), 'local failure');

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

    function value = asymmetricLoss(target, prediction)
        calls = calls + 1;
        difference = prediction - target;
        weights = ones(size(difference));
        weights(difference < 0) = 2;
        value = mean(weights .* difference .^ 2);
    end
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
