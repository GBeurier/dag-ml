classdef LocalImplementationRegistry < handle
    % Process-local registry for MATLAB loss and metric function handles.

    properties (Access = private)
        BindingId = 'binding:matlab'
        Entries
        NativeLibrary
    end

    methods
        function self = LocalImplementationRegistry(nativeLibrary)
            if nargin < 1
                nativeLibrary = getenv('DAGML_NATIVE_LIBRARY');
            end
            self.NativeLibrary = ...
                dagml.LocalImplementationRegistry.optionalScalarText( ...
                nativeLibrary, 'native library');
            self.Entries = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end

        function registerLoss(self, lossReference, implementation)
            self.registerImplementation(lossReference, implementation, 'loss');
        end

        function registerMetric(self, metricReference, implementation)
            self.registerImplementation(metricReference, implementation, 'metric');
        end

        function implementation = resolveLoss(self, lossReference)
            implementation = self.resolveImplementation(lossReference, 'loss');
        end

        function implementation = resolveTrainingLoss(self, trainingLossRole, phase)
            phase = dagml.LocalImplementationRegistry.scalarText(phase, 'training phase');
            if ~any(strcmp(phase, {'FIT_CV', 'REFIT'}))
                error('dagml:LocalImplementationRegistry:InvalidPhase', ...
                    'Training loss phase must be FIT_CV or REFIT.');
            end
            if ~dagml.LocalImplementationRegistry.roleApplies(trainingLossRole, phase)
                error('dagml:LocalImplementationRegistry:InactiveRole', ...
                    'Training loss role does not apply to phase %s.', phase);
            end
            implementation = self.resolveImplementation(trainingLossRole.loss, 'loss');
        end

        function implementation = resolveMetric(self, metricReference)
            implementation = self.resolveImplementation(metricReference, 'metric');
        end

        function value = invokeLoss(self, lossReference, varargin)
            implementation = self.resolveLoss(lossReference);
            value = implementation(varargin{:});
        end

        function value = invokeMetric(self, metricReference, varargin)
            implementation = self.resolveMetric(metricReference);
            value = implementation(varargin{:});
        end

        function [value, attestation] = invokeTrainingLoss(self, task, roleIndex, varargin)
            if nargin < 3 || isempty(roleIndex)
                roleIndex = 1;
            end
            if ~isnumeric(roleIndex) || ~isscalar(roleIndex) || ...
                    ~isfinite(roleIndex) || fix(roleIndex) ~= roleIndex || ...
                    roleIndex < 1 || roleIndex > flintmax
                error('dagml:LocalImplementationRegistry:RoleIndex', ...
                    'roleIndex must be a positive safe integer.');
            end
            if isempty(self.NativeLibrary)
                error('dagml:LocalImplementationRegistry:NativeLibrary', ...
                    ['Native DAG-ML library path is required; pass it to the ' ...
                    'constructor or set DAGML_NATIVE_LIBRARY.']);
            end
            taskJSON = dagml.LocalImplementationRegistry.nodeTaskJSON(task);
            [roleJSON, attestationJSON] = ...
                dagml.taskTrainingLossBindingNative( ...
                taskJSON, roleIndex - 1, self.NativeLibrary);
            role = jsondecode(roleJSON);
            requiredAttestation = jsondecode(attestationJSON);
            implementation = self.resolveImplementation(role.loss, 'loss');

            value = implementation(varargin{:});
            attestation = requiredAttestation;
        end

        function implementation = unregisterLoss(self, lossReference)
            implementation = self.unregisterImplementation(lossReference, 'loss');
        end

        function implementation = unregisterMetric(self, metricReference)
            implementation = self.unregisterImplementation(metricReference, 'metric');
        end

        function result = descriptors(self)
            registryKeys = sort(keys(self.Entries));
            result = cell(size(registryKeys));
            for index = 1:numel(registryKeys)
                entry = self.Entries(registryKeys{index});
                result{index} = entry.descriptor;
            end
        end

        function result = count(self)
            result = self.Entries.Count;
        end

        function clear(self)
            registryKeys = keys(self.Entries);
            if ~isempty(registryKeys)
                remove(self.Entries, registryKeys);
            end
        end

        function value = toJSON(~) %#ok<STOUT>
            error('dagml:LocalImplementationRegistry:Serialization', ...
                'DAG-ML local implementation registries cannot be serialized.');
        end

        function value = saveobj(~) %#ok<STOUT>
            error('dagml:LocalImplementationRegistry:Serialization', ...
                'DAG-ML local implementation registries cannot be serialized.');
        end

        function delete(self)
            self.clear();
        end
    end

    methods (Access = private)
        function registerImplementation(self, reference, implementation, semanticKind)
            if ~isa(implementation, 'function_handle')
                error('dagml:LocalImplementationRegistry:InvalidImplementation', ...
                    'Local loss and metric implementations must be function handles.');
            end
            [registryKey, descriptor] = self.validateDescriptor(reference, semanticKind);
            if isKey(self.Entries, registryKey)
                error('dagml:LocalImplementationRegistry:DuplicateKey', ...
                    'Duplicate local implementation registry key ''%s''.', registryKey);
            end
            self.Entries(registryKey) = struct( ...
                'descriptor', descriptor, 'implementation', implementation);
        end

        function implementation = resolveImplementation(self, reference, semanticKind)
            [registryKey, descriptor] = self.validateDescriptor(reference, semanticKind);
            if ~isKey(self.Entries, registryKey)
                error('dagml:LocalImplementationRegistry:MissingKey', ...
                    'Local implementation registry has no implementation for ''%s''.', ...
                    registryKey);
            end
            entry = self.Entries(registryKey);
            if ~dagml.LocalImplementationRegistry.sameJSONValue(entry.descriptor, descriptor)
                error('dagml:LocalImplementationRegistry:DescriptorMismatch', ...
                    ['Local implementation registered for ''%s'' does not match ' ...
                    'the requested descriptor.'], registryKey);
            end
            implementation = entry.implementation;
        end

        function implementation = unregisterImplementation(self, reference, semanticKind)
            implementation = self.resolveImplementation(reference, semanticKind);
            registryKey = dagml.LocalImplementationRegistry.scalarText( ...
                reference.implementation.registry_key, 'implementation registry_key');
            remove(self.Entries, registryKey);
        end

        function [registryKey, descriptor] = validateDescriptor(self, reference, semanticKind)
            if ~isstruct(reference) || ~isscalar(reference) || ...
                    ~isfield(reference, 'implementation') || ...
                    ~isstruct(reference.implementation) || ...
                    ~isscalar(reference.implementation)
                error('dagml:LocalImplementationRegistry:InvalidDescriptor', ...
                    'Implementation reference must contain an implementation descriptor.');
            end
            descriptor = reference.implementation;
            actualKind = dagml.LocalImplementationRegistry.requiredText( ...
                descriptor, 'semantic_kind', 'implementation semantic_kind');
            if ~strcmp(actualKind, semanticKind)
                error('dagml:LocalImplementationRegistry:SemanticKind', ...
                    'Expected a %s implementation descriptor, got %s.', ...
                    semanticKind, actualKind);
            end
            actualBinding = dagml.LocalImplementationRegistry.requiredText( ...
                descriptor, 'binding_id', 'implementation binding_id');
            if ~strcmp(actualBinding, self.BindingId)
                error('dagml:LocalImplementationRegistry:Binding', ...
                    'Local implementation requires binding_id ''%s'', got ''%s''.', ...
                    self.BindingId, actualBinding);
            end
            portability = dagml.LocalImplementationRegistry.requiredText( ...
                descriptor, 'portability', 'implementation portability');
            if ~any(strcmp(portability, {'host_local', 'portable_registered'}))
                error('dagml:LocalImplementationRegistry:Portability', ...
                    'Local implementation registry rejects portable_builtin descriptors.');
            end
            registryKey = dagml.LocalImplementationRegistry.requiredText( ...
                descriptor, 'registry_key', 'implementation registry_key');
            dagml.LocalImplementationRegistry.requiredText( ...
                descriptor, 'descriptor_fingerprint', ...
                'implementation descriptor_fingerprint');
        end
    end

    methods (Static, Access = private)
        function result = requiredText(value, field, label)
            if ~isstruct(value) || ~isscalar(value) || ~isfield(value, field)
                error('dagml:LocalImplementationRegistry:MissingField', ...
                    '%s must be present.', label);
            end
            result = dagml.LocalImplementationRegistry.scalarText(value.(field), label);
        end

        function result = scalarText(value, label)
            if ischar(value) && (isrow(value) || isempty(value))
                result = value;
            elseif isstring(value) && isscalar(value)
                result = char(value);
            else
                error('dagml:LocalImplementationRegistry:InvalidText', ...
                    '%s must be non-empty text.', label);
            end
            if isempty(strtrim(result))
                error('dagml:LocalImplementationRegistry:InvalidText', ...
                    '%s must be non-empty text.', label);
            end
        end

        function result = optionalScalarText(value, label)
            if ischar(value) && (isrow(value) || isempty(value))
                result = value;
            elseif isstring(value) && isscalar(value)
                result = char(value);
            else
                error('dagml:LocalImplementationRegistry:InvalidText', ...
                    '%s must be scalar text.', label);
            end
        end

        function result = nodeTaskJSON(task)
            result = dagml.LocalImplementationRegistry.scalarText( ...
                task, 'NodeTask JSON');
        end

        function result = roleApplies(role, phase)
            if ~isstruct(role) || ~isscalar(role) || ~isfield(role, 'phases')
                error('dagml:LocalImplementationRegistry:InvalidRole', ...
                    'Training loss role must be an object with phases.');
            end
            phases = role.phases;
            if ischar(phases)
                result = strcmp(phases, phase);
            elseif isstring(phases)
                result = any(strcmp(cellstr(phases(:)), phase));
            elseif iscell(phases)
                result = false;
                for index = 1:numel(phases)
                    item = dagml.LocalImplementationRegistry.scalarText( ...
                        phases{index}, 'training loss role phase');
                    result = result || strcmp(item, phase);
                end
            else
                error('dagml:LocalImplementationRegistry:InvalidRole', ...
                    'Training loss role phases must be text values.');
            end
        end

        function result = sameJSONValue(left, right)
            if isstruct(left) || isstruct(right)
                if ~isstruct(left) || ~isstruct(right) || ...
                        ~isequal(size(left), size(right))
                    result = false;
                    return;
                end
                leftFields = sort(fieldnames(left));
                rightFields = sort(fieldnames(right));
                if ~isequal(leftFields, rightFields)
                    result = false;
                    return;
                end
                result = true;
                for item = 1:numel(left)
                    for index = 1:numel(leftFields)
                        field = leftFields{index};
                        if ~dagml.LocalImplementationRegistry.sameJSONValue( ...
                                left(item).(field), right(item).(field))
                            result = false;
                            return;
                        end
                    end
                end
                return;
            end
            if iscell(left) || iscell(right)
                if ~iscell(left) || ~iscell(right) || ~isequal(size(left), size(right))
                    result = false;
                    return;
                end
                result = true;
                for index = 1:numel(left)
                    if ~dagml.LocalImplementationRegistry.sameJSONValue( ...
                            left{index}, right{index})
                        result = false;
                        return;
                    end
                end
                return;
            end
            if isnumeric(left) || isnumeric(right)
                result = isnumeric(left) && isnumeric(right) && ...
                    isequal(size(left), size(right)) && ...
                    all(isfinite(left(:))) && all(isfinite(right(:))) && ...
                    isequal(double(left), double(right));
                return;
            end
            if islogical(left) || islogical(right)
                result = islogical(left) && islogical(right) && isequal(left, right);
                return;
            end
            if (ischar(left) || isstring(left)) && ...
                    (ischar(right) || isstring(right))
                if (isstring(left) && ~isscalar(left)) || ...
                        (isstring(right) && ~isscalar(right))
                    result = false;
                else
                    result = strcmp(char(left), char(right));
                end
                return;
            end
            result = isequal(left, right);
        end
    end
end
