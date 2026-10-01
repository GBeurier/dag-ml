import type { Optimizer, OptimizerOptions, SearchAxis, TrialValue } from "@nirs4all/methods";

export interface N4mWasmHpoSnapshot {
  schema: "dagml.n4m.wasm-hpo.v1";
  contract: unknown;
  committed: Record<string, unknown> | null;
  prepared: Record<string, unknown> | null;
  n4mopt: number[];
}
export interface N4mWasmHpoOptions {
  Optimizer: typeof Optimizer;
  dagMl: { recover_host_hpo_checkpoint_json(checkpoint: string, prepared: string, interrupted: string): string };
  space: Record<string, SearchAxis>;
  options: OptimizerOptions;
  objective: Record<string, unknown>;
  /** Atomically persist both the native checkpoint and optimizer bytes. */
  persist(snapshot: N4mWasmHpoSnapshot): void;
  warmStart?: Record<string, TrialValue> | null;
  state?: N4mWasmHpoSnapshot | null;
}
/** Existing native Methods optimizer adapter, now shipped with the npm package. */
export declare class N4mWasmHostOptimizer {
  constructor(options: N4mWasmHpoOptions);
  readonly callback: (operation: string, payloadJson: string) => string;
  readonly committed: Record<string, unknown> | null;
  readonly prepared: Record<string, unknown> | null;
  snapshot(): N4mWasmHpoSnapshot;
  close(): void;
}
