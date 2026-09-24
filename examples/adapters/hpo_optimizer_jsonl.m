function hpo_optimizer_jsonl()
% Minimal MATLAB/Octave optimizer for dag-ml-cli run-host-hpo.
% Keep stdout strictly JSONL: one reply for each input line.
while true
    line = read_json_line();
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

function line = read_json_line()
% fgetl(0) can wait for EOF on a persistent Octave pipe.
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
