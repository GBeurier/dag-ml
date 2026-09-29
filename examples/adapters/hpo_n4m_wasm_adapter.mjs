/** Pair DAG-ML/WASM host HPO checkpoints with Methods JS/WASM N4MOPT state.
 *
 * `persist` must synchronously replace one durable record containing the full
 * snapshot. For a browser this can be a bounded localStorage record; callers
 * using asynchronous storage must first use an async DAG-ML host HPO API.
 */

function jsonValue(value) {
  if (typeof value === "bigint") {
    const number = Number(value);
    if (!Number.isSafeInteger(number)) throw new RangeError("DAG-ML JSON cannot represent this native integer exactly");
    return number;
  }
  if (Array.isArray(value)) return value.map(jsonValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, jsonValue(item)]));
  }
  if (typeof value === "number" && !Number.isFinite(value)) throw new RangeError("HPO parameter must be finite");
  return value;
}

function contractValue(value) {
  if (typeof value === "bigint") return { bigint: value.toString() };
  if (Array.isArray(value)) return value.map(contractValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, contractValue(value[key])]));
  }
  return value;
}

function same(left, right) {
  return JSON.stringify(contractValue(left)) === JSON.stringify(contractValue(right));
}

function terminalParts(terminal) {
  const evidence = terminal.evidence ?? terminal;
  return { index: evidence.trial_index, params: evidence.params,
    state: terminal.state, score: terminal.state === "complete" ? evidence.score : null };
}

function recordParams(record, space) {
  const params = {};
  for (const [name, axis] of Object.entries(space)) {
    if (axis.kind === "sorted_tuple") {
      if (Object.hasOwn(record.parameters, `${name}#0`)) {
        params[name] = Array.from({ length: axis.length }, (_, index) => jsonValue(record.parameters[`${name}#${index}`]));
      }
    } else if (Object.hasOwn(record.parameters, name)) {
      params[name] = jsonValue(record.parameters[name]);
    }
  }
  return params;
}

/** Owns one Methods optimizer and exposes a synchronous DAG-ML callback. */
export class N4mWasmHostOptimizer {
  constructor({ Optimizer, dagMl, space, options, objective, persist, warmStart = null, state = null }) {
    if (typeof persist !== "function") throw new TypeError("A synchronous persist(snapshot) callback is required");
    this.dagMl = dagMl;
    this.space = space;
    this.persist = persist;
    this.contract = contractValue({ space, options, objective, warmStart });
    this.committed = null;
    this.prepared = null;
    this.broken = false;
    if (state !== null) {
      if (state.schema !== "dagml.n4m.wasm-hpo.v1" || !same(state.contract, this.contract)
          || !Array.isArray(state.n4mopt) || state.n4mopt.length === 0) {
        throw new Error("Methods/WASM HPO snapshot contract mismatch");
      }
      this.optimizer = Optimizer.load(Uint8Array.from(state.n4mopt), space);
      this.committed = state.committed;
      this.prepared = state.prepared;
      try { this._recover(); }
      catch (error) { this.optimizer.dispose(); throw error; }
    } else {
      this.optimizer = Optimizer.create(space, options);
      try {
        if (warmStart !== null) this.optimizer.enqueue(warmStart);
        this._save();
      }
      catch (error) { this.optimizer.dispose(); throw error; }
    }
  }

  snapshot() {
    if (this.broken) throw new Error("Methods/WASM HPO session must be reloaded after a persistence failure");
    return { schema: "dagml.n4m.wasm-hpo.v1", contract: this.contract,
      committed: this.committed, prepared: this.prepared,
      n4mopt: Array.from(this.optimizer.save()) };
  }

  _save() {
    try {
      const result = this.persist(this.snapshot());
      if (result && typeof result.then === "function") throw new TypeError("HPO persistence must be synchronous");
    } catch (error) {
      this.broken = true;
      throw error;
    }
  }

  _history(checkpoint) {
    if (checkpoint === null) return;
    const records = this.optimizer.trialRecords();
    if (!Array.isArray(checkpoint.trials) || records.length < checkpoint.trials.length) {
      throw new Error("DAG-ML checkpoint has more terminal trials than Methods");
    }
    for (const [index, terminal] of checkpoint.trials.entries()) {
      const expected = terminalParts(terminal);
      const record = records[index];
      const status = { complete: "completed", pruned: "pruned", failed: "failed" }[expected.state];
      if (expected.index !== index || record.id !== BigInt(index) || record.status !== status
          || !same(recordParams(record, this.space), expected.params)
          || (status === "completed" && record.score !== expected.score)) {
        throw new Error(`DAG-ML and Methods terminal histories disagree at trial ${index}`);
      }
    }
  }

  _recover() {
    const records = this.optimizer.trialRecords();
    if (this.committed === null) {
      if (this.prepared !== null || records.length) throw new Error("Methods state exists without a DAG-ML checkpoint");
      this._save();
      return;
    }
    this._history(this.committed);
    let paired = this.committed.trials.length;
    if (this.prepared !== null) {
      const terminal = terminalParts(this.prepared.trials.at(-1));
      const record = records[paired];
      if (this.prepared.trials.length !== paired + 1 || terminal.index !== paired || !record
          || record.id !== BigInt(paired) || !same(recordParams(record, this.space), terminal.params)) {
        throw new Error("Prepared DAG-ML terminal differs from Methods trial");
      }
      if (record.status === "running") {
        if (terminal.state === "complete") this.optimizer.tell(record.id, "completed", terminal.score);
        else if (terminal.state === "failed") this.optimizer.tell(record.id, "failed", undefined, "recovered_prepared_failure");
        else throw new Error("Prepared pruned trial lacks its Methods pruning transition");
      }
      this._history(this.prepared);
      paired += 1;
    }
    const interrupted = [];
    for (const record of this.optimizer.trialRecords().slice(paired)) {
      if (record.id !== BigInt(paired + interrupted.length) || record.status !== "running") {
        throw new Error("Methods has an unpaired non-running trial");
      }
      interrupted.push({ trial_index: Number(record.id), params: recordParams(record, this.space) });
      this.optimizer.tell(record.id, "failed", undefined, "interrupted_before_native_evaluation");
    }
    const recovered = JSON.parse(this.dagMl.recover_host_hpo_checkpoint_json(
      JSON.stringify(this.committed), JSON.stringify(this.prepared), JSON.stringify(interrupted),
    ));
    this._history(recovered);
    this.committed = recovered;
    this.prepared = null;
    this._save();
  }

  /** Pass directly as the optimizer callback to `host_hpo_search_json`. */
  callback = (operation, payloadJson) => {
    if (this.broken) throw new Error("Methods/WASM HPO session must be reloaded");
    const payload = JSON.parse(payloadJson);
    let reply;
    if (operation === "ask") {
      const trial = this.optimizer.ask();
      if (trial.id !== BigInt(payload.trial_index)) throw new Error("DAG-ML and Methods trial IDs disagree");
      this._save();
      reply = { params: jsonValue(trial.parameters) };
    } else if (operation === "report_intermediate") {
      reply = { prune: this.optimizer.intermediate(payload.trial_index, payload.step, payload.score) };
      // `intermediate()` may terminalize the Methods trial as pruned. Keep the
      // durable state at its last RUNNING ask until DAG-ML prepares that terminal.
      // A crash here then recovers the orphan as failed, not as an unpaired prune.
    } else if (operation === "prepare_terminal") {
      this.prepared = payload.checkpoint;
      this._save();
      reply = { ok: true };
    } else if (operation === "tell") {
      this.optimizer.tell(payload.trial_index, "completed", payload.score);
      this._save();
      reply = { ok: true };
    } else if (operation === "pruned") {
      if (this.optimizer.trialRecords()[payload.trial_index]?.status !== "pruned") {
        throw new Error("DAG-ML pruned a trial that Methods did not prune");
      }
      reply = { ok: true };
    } else if (operation === "fail") {
      this.optimizer.tell(payload.trial_index, "failed", undefined, String(payload.error).slice(0, 200));
      this._save();
      reply = { ok: true };
    } else if (operation === "checkpoint") {
      this._history(payload.checkpoint);
      this.committed = payload.checkpoint;
      this.prepared = null;
      this._save();
      reply = { continue: true };
    } else {
      throw new Error(`Unsupported DAG-ML HPO operation: ${operation}`);
    }
    return JSON.stringify(reply);
  };

  close() { this.optimizer.dispose(); }
}
