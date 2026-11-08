/** Pair DAG-ML/WASM host HPO checkpoints with Methods JS/WASM N4MOPT state.
 *
 * `persist` must synchronously replace one durable record containing the full
 * snapshot. For a browser this can be a bounded localStorage record; callers
 * using asynchronous storage must first use an async DAG-ML host HPO API.
 */

import { strictJson } from "./n4m_multimodal_controller.mjs";

function binary64Token(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new RangeError("HPO float parameter must be finite");
  if (Object.is(value, -0)) return "-0.0";
  const token = JSON.stringify(value);
  return /[.e]/u.test(token) ? token : `${token}.0`;
}

function integerToken(value) {
  if (typeof value === "number" && (!Number.isInteger(value) || Math.abs(value) > 2 ** 53)) {
    throw new RangeError("HPO integer parameter must be exactly representable");
  }
  if (typeof value !== "number" && typeof value !== "bigint") throw new TypeError("HPO integer parameter must be an integer");
  return BigInt(value).toString();
}

function axisToken(axis, value) {
  if (axis.kind === "int" || axis.kind === "log_int") return integerToken(value);
  if (axis.kind === "float" || axis.kind === "log_float") return binary64Token(value);
  if (axis.kind === "ordinal") return Number.isSafeInteger(value) && !Object.is(value, -0) ? String(value) : binary64Token(value);
  if (axis.kind === "sorted_tuple") return `[${value.map(item => axis.integer ? integerToken(item) : binary64Token(item)).join(",")}]`;
  if (axis.type === "float") return binary64Token(value);
  if (axis.type === "integer") return integerToken(value);
  return JSON.stringify(value);
}

function tokens(parameters, space) {
  const members = [];
  for (const [name, axis] of Object.entries(space)) {
    if (axis.kind === "sorted_tuple") {
      const value = Object.hasOwn(parameters, name) ? parameters[name]
        : Object.hasOwn(parameters, `${name}#0`) ? Array.from({ length: axis.length }, (_, index) => parameters[`${name}#${index}`]) : null;
      if (value !== null) members.push(`${JSON.stringify(name)}:${axisToken(axis, value)}`);
    } else if (Object.hasOwn(parameters, name)) members.push(`${JSON.stringify(name)}:${axisToken(axis, parameters[name])}`);
  }
  return `{${members.join(",")}}`;
}

function hasUnpairedSurrogate(value) {
  return /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value);
}

function requireRepresentableSpace(space) {
  const names = new Set();
  for (const [name, axis] of Object.entries(space)) {
    if (hasUnpairedSurrogate(name)) throw new RangeError("HPO axis name must be well-formed UTF-16");
    const normalized = name.normalize("NFC");
    if (names.has(normalized)) throw new RangeError("HPO axis names collide after NFC normalization");
    names.add(normalized);
    if (axis.kind === "categorical" && axis.type === "string" && !axis.choices.every(choice => !hasUnpairedSurrogate(choice))) {
      throw new RangeError("HPO categorical choices must be well-formed UTF-16");
    }

  }
}

function jsonValue(value) {
  if (typeof value === "bigint") {
    const number = Number(value);
    if (!Number.isInteger(number) || Math.abs(number) > 2 ** 53) throw new RangeError("DAG-ML JSON cannot represent this native integer exactly");
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
  if (Object.is(value, -0)) return { negative_zero: true };
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
    requireRepresentableSpace(space);
    this.dagMl = dagMl;
    this.space = space;
    this.persist = persist;
    this.contract = contractValue({ space, options, objective, warmStart });
    this.committedJson = null;
    this.preparedJson = null;
    this.broken = false;
    if (state !== null) {
      if (state.schema !== "dagml.n4m.wasm-hpo.v2" || !same(state.contract, this.contract)
          || !Array.isArray(state.n4mopt) || state.n4mopt.length === 0
          || !(state.committed_json === null || typeof state.committed_json === "string")
          || !(state.prepared_json === null || typeof state.prepared_json === "string")) {
        throw new Error("Methods/WASM HPO snapshot contract mismatch");
      }
      this.optimizer = Optimizer.load(Uint8Array.from(state.n4mopt), space);
      this.committedJson = state.committed_json;
      this.preparedJson = state.prepared_json;
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

  get committed() { return this.committedJson === null ? null : JSON.parse(this.committedJson); }
  get prepared() { return this.preparedJson === null ? null : JSON.parse(this.preparedJson); }

  snapshot() {
    if (this.broken) throw new Error("Methods/WASM HPO session must be reloaded after a persistence failure");
    return { schema: "dagml.n4m.wasm-hpo.v2", contract: this.contract,
      committed_json: this.committedJson, prepared_json: this.preparedJson,
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
    if (this.committedJson === null) {
      if (this.preparedJson !== null || records.length) throw new Error("Methods state exists without a DAG-ML checkpoint");
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
      interrupted.push(`{"trial_index":${Number(record.id)},"params":${tokens(record.parameters, this.space)}}`);
      this.optimizer.tell(record.id, "failed", undefined, "interrupted_before_native_evaluation");
    }
    const recovered = this.dagMl.recover_host_hpo_checkpoint_json(
      this.committedJson, this.preparedJson ?? "null", `[${interrupted.join(",")}]`,
    );
    this._history(JSON.parse(recovered));
    this.committedJson = recovered;
    this.preparedJson = null;
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
      return `{"params":${tokens(trial.parameters, this.space)}}`;
    } else if (operation === "report_intermediate") {
      reply = { prune: this.optimizer.intermediate(payload.trial_index, payload.step, payload.score) };
      // `intermediate()` may terminalize the Methods trial as pruned. Keep the
      // durable state at its last RUNNING ask until DAG-ML prepares that terminal.
      // A crash here then recovers the orphan as failed, not as an unpaired prune.
    } else if (operation === "prepare_terminal") {
      this.preparedJson = strictJson(payloadJson, "checkpoint");
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
      this.committedJson = strictJson(payloadJson, "checkpoint");
      this.preparedJson = null;
      this._save();
      reply = { continue: true };
    } else {
      throw new Error(`Unsupported DAG-ML HPO operation: ${operation}`);
    }
    return JSON.stringify(reply);
  };

  close() { this.optimizer.dispose(); }
}
