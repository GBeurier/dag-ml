import type { RolePipeline } from "@nirs4all/methods";

export declare const N4M_PIPELINE_STEPS_PARAM: "pipeline_steps_v1";
export declare const N4M_PIPELINE_STEPS_CONTROLLER_VERSION: "1.1.0";
export declare const N4M_PIPELINE_STEPS_PROFILE: "pipeline_steps.v1";
export type MethodsRecipeStep =
  | { class: string; methodId?: never; params?: Record<string, unknown> }
  | { methodId: string; class?: never; params?: Record<string, unknown> };
/** Reserved node parameter; opt into controllerVersion 1.1.0. */
export interface N4mPipelineStepsV1Params {
  pipeline_steps_v1: MethodsRecipeStep[];
  [modelParameter: string]: unknown;
}

export interface HostMatrix {
  data: Float64Array;
  rows: number;
  cols: number;
}
export interface HostRows {
  sampleIds: string[];
  matrix: HostMatrix;
  featureNames?: string[];
}
export interface HostView {
  sample_ids: string[];
  partition: string;
  source_ids?: string[];
  [key: string]: unknown;
}
export interface MethodsRegressionControllerOptions {
  methods: { RolePipeline: typeof RolePipeline };
  /** Operators from the compiled graph; NodeTask intentionally has no operator object. */
  operators: Record<string, { type: string; steps?: Array<{ class?: string; methodId?: string; params?: Record<string, unknown> }> }>;
  resolveFeatures(request: { key: string; view: HostView; task: Record<string, unknown> }): HostRows;
  resolveTargets(request: { sampleIds: string[]; targetNames: string[]; task: Record<string, unknown> }): HostRows;
  targetNames?: string[];
  controllerId?: string;
  /** Default 1.0.0; pipeline_steps_v1 requires 1.1.0 and payload profile v2. */
  controllerVersion?: string;
  /** Synchronous lowercase SHA-256; required for REFIT and portable hydration. */
  digest?: ((bytes: Uint8Array) => string) | null;
}
export interface ControllerModelHandle {
  handle: number;
  kind: "model";
  owner_controller: string;
}
export interface ControllerArtifactRequest {
  controller_id: string;
  node_id: string;
  params_fingerprint: string;
  artifact: { id: string; content_fingerprint: string; [key: string]: unknown };
}
/** Numerical buffers and model ownership are host-local; CV/OOF/scoring are native. */
export declare class N4mWasmRegressionController {
  constructor(options: MethodsRegressionControllerOptions);
  readonly controllerId: string;
  readonly controllerVersion: string;
  readonly callback: (controllerId: string, taskJson: string, exactSeed?: string | null) => string;
  manifest(dagMl: { derive_controller_manifest_json(json: string): string }): Record<string, unknown>;
  invoke(controllerId: string, taskJson: string): string;
  artifactPayload(artifactId: string): Uint8Array;
  hydrate(request: ControllerArtifactRequest, payload: Uint8Array): ControllerModelHandle;
  close(): void;
}
