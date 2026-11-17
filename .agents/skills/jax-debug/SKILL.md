---
name: jax-debug
description: Debug JAX errors including tracing issues, jit/vmap/pmap shape problems, PRNG key bugs, dtype promotion, device placement, NaN/Inf, and OOM. Use when the user hits a JAX runtime/trace error or wants help tracing a bug.
---

# JAX Debug Assistant

Help the user diagnose and fix JAX errors systematically. JAX's
trace-then-compile model means most bugs surface at trace time, with stack
traces that point at JAX internals rather than user code. Debugging requires
different reflexes than PyTorch.

## Debugging Method: Trace First, Then Step Inside the Compile

1. **Reproduce eagerly first.** Disable JIT (`jax.config.update("jax_disable_jit", True)`)
   to confirm the bug isn't an artifact of tracing. If it disappears, the bug
   is compile-time-only (recompilation, abstract-tracer leakage).
2. **Identify the abstraction level of the offending op.** Is the error from
   eager execution, a `jit` trace, a `pmap`/`vmap` transformation, or
   gradient computation? Each has different failure modes.
3. **Print at trace time vs runtime.** `print(x)` shows the *tracer*
   (`Traced<ShapedArray(...)>`). Useful for shape/dtype. `jax.debug.print("{}", x)`
   shows the *value at runtime*. Useful for content.
4. **Identify the divergence**: where does actual vs expected dtype, shape,
   sharding, or value first differ?

## Common Issues

### Tracing Errors (`ConcretizationTypeError`, `TracerArrayConversionError`)

The single most common JAX bug: code uses a traced array as if it were a
concrete value (e.g., `if x > 0`, `int(x)`, `x.item()`, `np.array(x)` inside
a jitted function).

- **Convert Python control flow to JAX control flow.** `if` → `jax.lax.cond`;
  `for` over array → `jax.lax.scan` or `jax.lax.fori_loop`; early exit → not
  available, restructure.
- **Mark Python-side branching args as `static_argnums` / `static_argnames`**
  in `jax.jit`. But beware. Each static value triggers a recompile.
- **`.item()` / `int()` / `bool()` on tracers is never going to work inside
  `jit`.** Move that op outside the jitted region.

### Shape Errors Inside `vmap` / `pmap`

- `vmap` adds a batch dim; check that broadcasted ops still align after
  vmapping. `in_axes` controls which dims of which inputs to map over.
  default `0` is rarely what you want with kwargs.
- `pmap` requires the leading dim to equal `jax.local_device_count()`. Reshape
  with `x.reshape((n_devices, -1, *rest))` before `pmap`.
- Use `jax.tree.map(lambda x: x.shape, params)` to inspect a PyTree of shapes.

### PRNG Key Management

JAX RNG is **explicit**. The two failure modes:

- **Reusing a key.** Two calls with the same key give identical "random"
  output. Silent reproducibility bug. Always `split` before consuming.
- **Forgetting to thread the key through `scan` / `vmap`.** Each iteration
  needs its own split key. Pattern:
  ```python
  keys = jax.random.split(key, num_steps)
  jax.lax.scan(lambda c, k: step(c, k), init, keys)
  ```

### Dtype Promotion & x64

- JAX defaults to **float32 everywhere**. Even literals like `1.0`. To use
  float64: `jax.config.update("jax_enable_x64", True)` *before any array
  creation*. Setting it after silently fails.
- "Weak" types (Python scalars) promote toward the array dtype, not vice
  versa. `jnp.array([1.0], dtype=jnp.bfloat16) * 1.0` stays bf16, which is
  often surprising.
- Mixed-precision: under `jmp` or manual `lax.precision`, matmuls can drop to
  bf16 silently. Check with `jax.make_jaxpr(fn)(...)` to inspect the
  lowered ops.

### Device Placement & Sharding

- Default device is set by `JAX_PLATFORMS` env var (`cpu`, `gpu`, `tpu`) or
  `jax.config.update("jax_platforms", ...)`.
- `jax.device_put(x, device)` for explicit placement.
- For SPMD: use `jax.sharding.NamedSharding` + `jax.jit(..., in_shardings=..., out_shardings=...)`. Shape mismatches at shard boundaries error with cryptic
  XLA messages. Print `x.sharding` to inspect.

### NaN / Inf Debugging

- **Best tool:** `jax.config.update("jax_debug_nans", True)` raises at the op
  that produces the first NaN. Slows things down. Turn off for production.
- `jax.config.update("jax_debug_infs", True)` for Infs.
- Common producers: `log(0)`, `sqrt(negative)`, `0/0` in softmax stabilization,
  unstable layer norm, gradient through `where(cond, x, 1/x)` (the false
  branch is still traced).

### Recompilation Surprises

JAX recompiles whenever a static argument changes OR an array's
shape/dtype/sharding changes.

- **Check with `jax.jit`'s `_cache_size()`** or enable
  `JAX_LOG_COMPILES=1` to see every compile.
- Recompiling on every batch usually means a shape is leaking through
  `static_argnames`, or you're passing a Python int that should be a JAX array.
- Cache only persists within a process. Use `jax.experimental.compilation_cache`
  for cross-process caching.

### Gradient Issues

- `jax.grad` operates only on the *first* argument by default. Use `argnums`
  or `argnums=(...)` for multiple.
- `stop_gradient` is the JAX equivalent of `.detach()`.
- `value_and_grad` returns `(value, grad)`. Useful for logging without a
  second forward pass.
- Custom VJPs (`@jax.custom_vjp`) silently override autodiff; if a gradient
  looks wrong, grep for `custom_vjp` / `custom_jvp` on the call path.

### OOM

- JAX preallocates **75% of GPU memory by default**. Disable with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, or cap with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`.
- Per-host memory: `XLA_PYTHON_CLIENT_ALLOCATOR=platform` switches to BFC.
- For long traces, the compiled HLO can itself be huge. Check with
  `jax.jit(fn).lower(...).compile().memory_analysis()`.

## Diagnostic Snippets

```python
# Inspect a tracer at trace time
print(f"x: shape={x.shape}, dtype={x.dtype}")  # works on tracers

# Inspect a value at runtime (inside jit)
jax.debug.print("x = {}", x)

# Inspect a PyTree of shapes
jax.tree.map(lambda t: t.shape, params)

# See the lowered IR
print(jax.make_jaxpr(fn)(*example_inputs))

# Check sharding
print(x.sharding, x.device)

# Enable NaN-on-error globally for a debug session
jax.config.update("jax_debug_nans", True)
```

## When to Bail Out and Run Eagerly

If a `jit`-related error is impenetrable after 10 minutes, drop into eager
mode (`jax.disable_jit()` context manager or the global config) and step
through with pdb. Once the logic is correct eagerly, re-enable `jit`.

## Cross-references

- `/pytorch-debug`. Sibling skill for PyTorch.
- `/distributed`. For multi-device JAX (`pjit`, `Mesh`, `pmap` at scale).
- `/experiment`. JAX experiments often use `flax` + `optax`; see profiling
  / checkpointing sections.

## Scope

$ARGUMENTS
