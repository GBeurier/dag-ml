function hpo_optimizer_jsonl()
% Minimal MATLAB/Octave optimizer for dag-ml-cli run-host-hpo.
% Keep stdout strictly JSONL: one reply for each input line.
while true
    line = fgetl(0);
    if ~ischar(line)
        break;
    end
    event = jsondecode(line);
    operation = char(event.operation);
    if strcmp(operation, 'init')
        reply = '{"prepared_checkpoint":null,"interrupted":[]}';
    elseif strcmp(operation, 'ask')
        reply = sprintf('{"params":{"n_components":%d}}', event.trial_index + 1);
    elseif strcmp(operation, 'report_intermediate')
        reply = '{"prune":false}';
    elseif any(strcmp(operation, {'tell', 'pruned', 'fail', 'prepare_terminal', 'checkpoint'}))
        reply = '{"ok":true}';
    else
        reply = jsonencode(struct('error', ['unsupported HPO operation ' operation]));
    end
    fprintf(1, '%s\n', reply);
    if exist('OCTAVE_VERSION', 'builtin')
        fflush(1);
    else
        drawnow;
    end
end
end
