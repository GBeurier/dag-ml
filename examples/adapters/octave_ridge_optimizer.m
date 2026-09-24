function octave_ridge_optimizer()
% Real Octave optimizer: candidate 3 is pruned after its first fold.
while true
    line = read_json_line();
    if ~ischar(line), break; end
    event = jsondecode(line);
    switch char(event.operation)
        case 'init'
            reply = '{"prepared_checkpoint":null,"interrupted":[]}';
        case 'ask'
            reply = jsonencode(struct('params', struct('n_components', event.trial_index + 1)));
        case 'report_intermediate'
            prune = event.trial_index == 2 && event.step == 0;
            reply = jsonencode(struct('prune', prune));
        case {'tell', 'pruned', 'fail', 'prepare_terminal', 'checkpoint'}
            reply = '{"ok":true}';
        otherwise
            reply = jsonencode(struct('error', ['unsupported HPO operation ' char(event.operation)]));
    end
    fprintf(1, '%s\n', reply);
    fflush(1);
end
end

function line = read_json_line()
% Octave fgetl(0) waits for pipe EOF when the writer keeps stdin open.
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
