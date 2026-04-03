/* @ts-self-types="./wasm_agent.d.ts" */

/**
 * WASM-exported PPO agent that wraps the game simulation and neural network.
 */
export class WasmAgent {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmAgentFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmagent_free(ptr, 0);
    }
    /**
     * Export current PPO weights as JSON.
     * @returns {string}
     */
    export_weights() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmagent_export_weights(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get current agent stats.
     * @returns {any}
     */
    get_stats() {
        const ret = wasm.wasmagent_get_stats(this.__wbg_ptr);
        return ret;
    }
    /**
     * Load DQN weights (partial, shared layers only) from the model_weights.json format.
     * @param {string} json
     * @returns {boolean}
     */
    load_dqn_weights(json) {
        const ptr0 = passStringToWasm0(json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmagent_load_dqn_weights(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
    /**
     * Load PPO weights from JSON string.
     * @param {string} json
     * @returns {boolean}
     */
    load_weights(json) {
        const ptr0 = passStringToWasm0(json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmagent_load_weights(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
    /**
     * Create a new WasmAgent. Pass a JSON config string or empty/null for defaults.
     * @param {string} config_json
     */
    constructor(config_json) {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmagent_new(ptr0, len0);
        this.__wbg_ptr = ret >>> 0;
        WasmAgentFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Reset game and agent state for a new episode.
     */
    reset() {
        wasm.wasmagent_reset(this.__wbg_ptr);
    }
    /**
     * Enable or disable learning (training vs. inference-only mode).
     * @param {boolean} enabled
     */
    set_learning(enabled) {
        wasm.wasmagent_set_learning(this.__wbg_ptr, enabled);
    }
    /**
     * Main loop step: get state, select action, step game, store transition, return render data.
     * @returns {any}
     */
    step() {
        const ret = wasm.wasmagent_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Turbo mode: run N steps without building render data. Returns stats.
     * @param {number} n
     * @returns {any}
     */
    train_steps(n) {
        const ret = wasm.wasmagent_train_steps(this.__wbg_ptr, n);
        return ret;
    }
}
if (Symbol.dispose) WasmAgent.prototype[Symbol.dispose] = WasmAgent.prototype.free;

export class WasmGame {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmGameFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmgame_free(ptr, 0);
    }
    /**
     * Lookahead: simulate each action forward n_steps.
     * Returns 0 for safe actions, -50 for life loss, -100 for game over.
     * Only overrides policy when an action leads to death.
     * @param {Uint8Array} actions
     * @param {number} n_steps
     * @returns {Float32Array}
     */
    evaluate_actions(actions, n_steps) {
        const ptr0 = passArray8ToWasm0(actions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmgame_evaluate_actions(this.__wbg_ptr, ptr0, len0, n_steps);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Get the 50-feature state vector for AI inference.
     * @returns {Float32Array}
     */
    get_state() {
        const ret = wasm.wasmgame_get_state(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Create a new WasmGame instance with a default seed.
     */
    constructor() {
        const ret = wasm.wasmgame_new();
        this.__wbg_ptr = ret >>> 0;
        WasmGameFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Reset for a new game. Returns the initial state vector.
     */
    reset() {
        wasm.wasmgame_reset(this.__wbg_ptr);
    }
    /**
     * Reset to a specific level (for curriculum training).
     * @param {number} level
     */
    reset_at_level(level) {
        wasm.wasmgame_reset_at_level(this.__wbg_ptr, level);
    }
    /**
     * Tick the game with an action code (0-5). Returns full render state as a JS object.
     * @param {number} dt
     * @param {number} action
     * @returns {any}
     */
    tick(dt, action) {
        const ret = wasm.wasmgame_tick(this.__wbg_ptr, dt, action);
        return ret;
    }
    /**
     * Tick with raw keyboard input booleans. Returns full render state as a JS object.
     * @param {number} _dt
     * @param {boolean} left
     * @param {boolean} right
     * @param {boolean} fire
     * @returns {any}
     */
    tick_input(_dt, left, right, fire) {
        const ret = wasm.wasmgame_tick_input(this.__wbg_ptr, _dt, left, right, fire);
        return ret;
    }
}
if (Symbol.dispose) WasmGame.prototype[Symbol.dispose] = WasmGame.prototype.free;

export class WasmOnlineDQN {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmOnlineDQNFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmonlinedqn_free(ptr, 0);
    }
    /**
     * Export current policy network weights as JSON.
     * @returns {string}
     */
    export_weights() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmonlinedqn_export_weights(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get Q-values for an external state (for overlay display).
     * @param {Float32Array} state
     * @returns {Float32Array}
     */
    get_q_values(state) {
        const ptr0 = passArrayF32ToWasm0(state, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmonlinedqn_get_q_values(this.__wbg_ptr, ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Get current training statistics.
     * @returns {any}
     */
    get_stats() {
        const ret = wasm.wasmonlinedqn_get_stats(this.__wbg_ptr);
        return ret;
    }
    /**
     * Load pretrained weights from JSON (model_weights.json format).
     * Initializes both policy and target networks.
     * @param {string} json
     * @returns {boolean}
     */
    load_weights(json) {
        const ptr0 = passStringToWasm0(json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmonlinedqn_load_weights(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
    /**
     * Create a new online DQN learner with default config.
     */
    constructor() {
        const ret = wasm.wasmonlinedqn_new();
        this.__wbg_ptr = ret >>> 0;
        WasmOnlineDQNFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Pick the best action for an external state (e.g. from the visible game).
     * State should be the raw features (54), will be frame-stacked internally.
     * @param {Float32Array} state
     * @returns {number}
     */
    pick_action(state) {
        const ptr0 = passArrayF32ToWasm0(state, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmonlinedqn_pick_action(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * Set the exploration epsilon.
     * @param {number} eps
     */
    set_epsilon(eps) {
        wasm.wasmonlinedqn_set_epsilon(this.__wbg_ptr, eps);
    }
    /**
     * Set the learning rate.
     * @param {number} lr
     */
    set_lr(lr) {
        wasm.wasmonlinedqn_set_lr(this.__wbg_ptr, lr);
    }
    /**
     * Run one game step: observe, act, store transition, maybe train.
     * @returns {any}
     */
    step() {
        const ret = wasm.wasmonlinedqn_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Run N steps in turbo mode (no per-step JS object overhead).
     * @param {number} n
     * @returns {any}
     */
    train_steps(n) {
        const ret = wasm.wasmonlinedqn_train_steps(this.__wbg_ptr, n);
        return ret;
    }
}
if (Symbol.dispose) WasmOnlineDQN.prototype[Symbol.dispose] = WasmOnlineDQN.prototype.free;

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_5549492daedad139: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_new_4370be21fa2b2f80: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_48e1d86cfd30c8e7: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_from_slice_170b9484b744c862: function(arg0, arg1) {
            const ret = new Float32Array(getArrayF32FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_push_d0006a37f9fcda6d: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_set_991082a7a49971cf: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./wasm_agent_bg.js": import0,
    };
}

const WasmAgentFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmagent_free(ptr >>> 0, 1));
const WasmGameFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmgame_free(ptr >>> 0, 1));
const WasmOnlineDQNFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmonlinedqn_free(ptr >>> 0, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('wasm_agent_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
