/* tslint:disable */
/* eslint-disable */

/**
 * WASM-exported PPO agent that wraps the game simulation and neural network.
 */
export class WasmAgent {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Export current PPO weights as JSON.
     */
    export_weights(): string;
    /**
     * Get current agent stats.
     */
    get_stats(): any;
    /**
     * Load DQN weights (partial, shared layers only) from the model_weights.json format.
     */
    load_dqn_weights(json: string): boolean;
    /**
     * Load PPO weights from JSON string.
     */
    load_weights(json: string): boolean;
    /**
     * Create a new WasmAgent. Pass a JSON config string or empty/null for defaults.
     */
    constructor(config_json: string);
    /**
     * Reset game and agent state for a new episode.
     */
    reset(): void;
    /**
     * Enable or disable learning (training vs. inference-only mode).
     */
    set_learning(enabled: boolean): void;
    /**
     * Main loop step: get state, select action, step game, store transition, return render data.
     */
    step(): any;
    /**
     * Turbo mode: run N steps without building render data. Returns stats.
     */
    train_steps(n: number): any;
}

export class WasmGame {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Lookahead: simulate each action forward n_steps.
     * Returns 0 for safe actions, -50 for life loss, -100 for game over.
     * Only overrides policy when an action leads to death.
     */
    evaluate_actions(actions: Uint8Array, n_steps: number): Float32Array;
    /**
     * Get the 50-feature state vector for AI inference.
     */
    get_state(): Float32Array;
    /**
     * Create a new WasmGame instance with a default seed.
     */
    constructor();
    /**
     * Reset for a new game. Returns the initial state vector.
     */
    reset(): void;
    /**
     * Reset to a specific level (for curriculum training).
     */
    reset_at_level(level: number): void;
    /**
     * Tick the game with an action code (0-5). Returns full render state as a JS object.
     */
    tick(dt: number, action: number): any;
    /**
     * Tick with raw keyboard input booleans. Returns full render state as a JS object.
     */
    tick_input(_dt: number, left: boolean, right: boolean, fire: boolean): any;
}

export class WasmOnlineDQN {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Export current policy network weights as JSON.
     */
    export_weights(): string;
    /**
     * Get Q-values for an external state (for overlay display).
     */
    get_q_values(state: Float32Array): Float32Array;
    /**
     * Get current training statistics.
     */
    get_stats(): any;
    /**
     * Load pretrained weights from JSON (model_weights.json format).
     * Initializes both policy and target networks.
     */
    load_weights(json: string): boolean;
    /**
     * Create a new online DQN learner with default config.
     */
    constructor();
    /**
     * Pick the best action for an external state (e.g. from the visible game).
     * State should be the raw features (54), will be frame-stacked internally.
     */
    pick_action(state: Float32Array): number;
    /**
     * Set the exploration epsilon.
     */
    set_epsilon(eps: number): void;
    /**
     * Set the learning rate.
     */
    set_lr(lr: number): void;
    /**
     * Run one game step: observe, act, store transition, maybe train.
     */
    step(): any;
    /**
     * Run N steps in turbo mode (no per-step JS object overhead).
     */
    train_steps(n: number): any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmagent_free: (a: number, b: number) => void;
    readonly __wbg_wasmgame_free: (a: number, b: number) => void;
    readonly __wbg_wasmonlinedqn_free: (a: number, b: number) => void;
    readonly wasmagent_export_weights: (a: number) => [number, number];
    readonly wasmagent_get_stats: (a: number) => any;
    readonly wasmagent_load_dqn_weights: (a: number, b: number, c: number) => number;
    readonly wasmagent_load_weights: (a: number, b: number, c: number) => number;
    readonly wasmagent_new: (a: number, b: number) => number;
    readonly wasmagent_reset: (a: number) => void;
    readonly wasmagent_set_learning: (a: number, b: number) => void;
    readonly wasmagent_step: (a: number) => any;
    readonly wasmagent_train_steps: (a: number, b: number) => any;
    readonly wasmgame_evaluate_actions: (a: number, b: number, c: number, d: number) => [number, number];
    readonly wasmgame_get_state: (a: number) => [number, number];
    readonly wasmgame_new: () => number;
    readonly wasmgame_reset: (a: number) => void;
    readonly wasmgame_reset_at_level: (a: number, b: number) => void;
    readonly wasmgame_tick: (a: number, b: number, c: number) => any;
    readonly wasmgame_tick_input: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly wasmonlinedqn_export_weights: (a: number) => [number, number];
    readonly wasmonlinedqn_get_q_values: (a: number, b: number, c: number) => [number, number];
    readonly wasmonlinedqn_get_stats: (a: number) => any;
    readonly wasmonlinedqn_load_weights: (a: number, b: number, c: number) => number;
    readonly wasmonlinedqn_new: () => number;
    readonly wasmonlinedqn_pick_action: (a: number, b: number, c: number) => number;
    readonly wasmonlinedqn_set_epsilon: (a: number, b: number) => void;
    readonly wasmonlinedqn_set_lr: (a: number, b: number) => void;
    readonly wasmonlinedqn_step: (a: number) => any;
    readonly wasmonlinedqn_train_steps: (a: number, b: number) => any;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
