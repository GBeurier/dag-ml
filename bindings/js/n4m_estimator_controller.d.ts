/** Native Methods role registration; all fitting and prediction stay in WASM. */
export interface EstimatorRoleMatrix { data: Float64Array; rows: number; cols: number; }
export interface EstimatorFeatureResolution { sampleIds: string[]; matrix: EstimatorRoleMatrix; }
export interface EstimatorControllerOptions {
  methods: object;
  dagMl: object;
  digest: (bytes: Uint8Array) => string;
  targetNames: string[];
  resolveFeatures: (request: { key: string; view: { sample_ids: string[]; source_ids: string[]; partition: string }; task: Record<string, unknown> }) => EstimatorFeatureResolution;
  resolveTargets: (request: { sampleIds: string[]; task: Record<string, unknown> }) => EstimatorFeatureResolution;
}
export declare class N4mWasmEstimatorControllers {
  constructor(options: EstimatorControllerOptions);
  readonly manifests: Record<string, unknown>[];
  readonly callback: (owner: string, taskJson: string) => string;
  planned(node: Record<string, unknown>): Record<string, unknown>;
  close(): void;
}
