import type { HostRows, ControllerArtifactRequest, ControllerModelHandle } from './n4m_controller.d.ts';
export declare const N4M_SOURCE_PROFILE: 'source_composition.v1';
export declare const N4M_SOURCE_CONTROLLER_VERSION: '1.0.0';
export declare const N4M_SOURCE_RECIPE_PARAM: 'source_composition_v1';
export interface SourceOperation { source_id: string; steps: Array<{methodId: string; params?: Record<string, unknown>}> }
export interface SourceSchema { representation: string; input_shape: number[]; feature_names: string[]; identity: string }
export interface SourcePipelineOperator {
  type: 'N4mSourcePipeline'; sourceOps: SourceOperation[]; sourceSchemas: Record<string, SourceSchema>;
  steps: Array<{methodId: string; params?: Record<string, unknown>}>; hybrid: boolean;
}
export interface SourceControllerOptions {
  methods: { RolePipeline: any; NativeEstimator: any; methodClass(methodId: string): any;
    manifest(): {methods: Array<Record<string, any>>} };
  operators: Record<string, SourcePipelineOperator>;
  resolveSource(request: {sourceId: string; sampleIds: string[]; task: Record<string, any>}): HostRows;
  resolveTargets(request: {sampleIds: string[]; targetNames: string[]; task: Record<string, any>}): HostRows;
  targetNames?: string[]; controllerId?: string; digest(bytes: Uint8Array): string;
  onTask?: ((task: {nodeId: string; phase: string; foldId: string | null}) => void) | null;
  /** Default 'regression'. Classification needs one target whose values are class ids 0..k-1. */
  task?: 'regression' | 'classification';
  /** Classification only: ordered vocabulary; class id i is classLabels[i]. */
  classLabels?: string[] | null;
}
export declare class N4mWasmSourceController {
  constructor(options: SourceControllerOptions);
  readonly controllerId: string; readonly controllerVersion: string;
  readonly callback: (controllerId: string, taskJson: string, exactSeed?: string | null) => string;
  readonly audit: Array<Record<string, unknown>>;
  readonly task: 'regression' | 'classification';
  readonly classification: boolean;
  manifest(dagMl: {derive_controller_manifest_json(json: string): string}): Record<string, unknown>;
  artifactPayload(id: string): Uint8Array;
  hydrate(request: ControllerArtifactRequest, payload: Uint8Array): ControllerModelHandle;
  invoke(controllerId: string, taskJson: string, exactSeed?: string | null): string;
  close(): void;
}
