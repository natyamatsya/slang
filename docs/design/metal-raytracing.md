# Metal Ray-Tracing Pipeline Stages — Implementation Specification

Status: **P0 + P1 + P2 implemented** (all six stages; `TraceRay` and
`CallShader` from raygen; see §10 for the implementation notes and the
exact places where the implementation deviates from the text below)
Target: `-target metal` support for the six ray-tracing pipeline stages
(`raygeneration`, `miss`, `closesthit`, `anyhit`, `intersection`, `callable`).

## 1. Summary

Slang currently rejects ray-tracing pipeline entry points for the Metal target:

```
error 36107: entrypoint 'rayGenMain' uses features that are not available in
'raygen' stage for 'metal' compilation target.
```

The gate is the `raytracing` capability alias (`source/slang/slang-capabilities.capdef:1347`):

```
alias raytracing = GL_EXT_ray_tracing | _sm_6_3 | cuda;
```

Metal has no driver-level ray-tracing pipeline (no SBT object, no
raygen/miss/hit dispatch by the driver), but it has all the primitives to
implement one: a compute kernel driving `metal::raytracing::intersector`,
**visible function tables** (`MTLVisibleFunctionTable`, function pointers,
`supportsFunctionPointers`, MSL ≥ 2.3) and **intersection function tables**
(`MTLIntersectionFunctionTable`, consumed natively by the intersector). This
document specifies how Slang lowers the DXR-style programming model onto those
primitives, so that a single Slang RT code base compiles to D3D12/Vulkan/OptiX
*and* Metal.

There is precedent in-repo for lowering RT stages to a non-DXR execution model:
the CUDA/OptiX backend (`slang-emit-cuda.cpp:494`, `CASE(RayGeneration,
__raygen__)`, `TraceRay` → `optixTrace`). Metal follows the same shape — the
stage semantics live in the compiler, the target supplies an unconventional
runtime — with the extra twist that Metal's "pipeline" is assembled by the
*application* from linked functions, which makes the **ABI contract** (§4) the
core of this spec. The inline half of Metal ray tracing (`RayQuery` →
`raytracing::intersection_query`) already works (`slang-emit-metal.cpp:1316`);
this spec adds the pipeline half.

## 2. Execution model mapping

| DXR / Slang concept | Metal lowering |
|---|---|
| raygen entry point | the `[[kernel]]` compute function; launch grid = `DispatchRays` dimensions |
| `TraceRay()` | inline `intersector<...>::intersect()` + an indexed call through the visible function table (§5) |
| miss / closesthit / callable entry points | `[[visible]]` functions with **one uniform signature** (§4.2), registered in a `MTLVisibleFunctionTable` |
| anyhit / intersection entry points | `[[intersection(triangle|bounding_box, instancing, ...)]]` functions in a `MTLIntersectionFunctionTable`; the intersector invokes them natively |
| shader binding table | a small device buffer of **table indices** (region bases + per-record indices) provided by the application (§4.4); geometry→hit-group selection uses the DXR index formula (§5.2) |
| `DispatchRays(w,h,d)` | `dispatchThreads(MTLSize(w,h,d), ...)` on the linked pipeline |
| pipeline object | `MTLComputePipelineDescriptor` with `linkedFunctions` = all stage functions; tables created from the pipeline state |

Required device features: `supportsRaytracing()` **and**
`supportsFunctionPointers()`; MSL language version ≥ 2.4 (matches what the
existing intersection-query emission already requires).

## 3. Capability changes

`source/slang/slang-capabilities.capdef`:

- Extend the alias:
  `alias raytracing = GL_EXT_ray_tracing | _sm_6_3 | cuda | metallib_2_4;`
  This automatically opens the stage aliases (`alias raygen = _raygen +
  raytracing;` etc., `:1478ff`) for the Metal target.
- Features that cannot be mapped (e.g. shader execution reordering, `ser`)
  keep their existing, narrower aliases and continue to error on Metal.

## 4. The ABI contract

RT stages are compiled separately (possibly in separate `MTLLibrary` objects —
`MTLLinkedFunctions` accepts functions from any library), and the application
assembles the pipeline. Everything below is therefore a **stable, documented
convention** that both Slang codegen and the application runtime implement.

### 4.1 The ray context

All cross-stage communication goes through one thread-local struct owned by the
kernel and passed by reference:

```metal
struct slang_RTContext
{
    // the ray for the *current* trace (world space)
    float3 origin;    float tMin;
    float3 direction; float tMax;
    uint   rayFlags;
    // committed-hit state, filled by the kernel after intersect()
    float  hitT;
    uint   instanceIndex;   // and instanceID (user), geometryIndex
    uint   instanceID;
    uint   geometryIndex;
    uint   primitiveIndex;
    uint   hitKind;
    float2 triBarycentrics;             // triangle hits
    uchar  attributes[SLANG_RT_MAX_ATTRIBUTE_SIZE]; // procedural hits (default 32, DXR parity)
    // payload blob for the current trace
    uchar  payload[SLANG_RT_MAX_PAYLOAD_SIZE];
    // launch state
    uint3  launchIndex;
    uint3  launchDim;
};
```

`SLANG_RT_MAX_PAYLOAD_SIZE` defaults to 64 bytes and is overridable by a
compiler option (`-metal-rt-max-payload-size N`); exceeding it with a concrete
payload type is a compile-time diagnostic. This blob approach is what DXR
drivers do internally (payload registers) and is what makes a *single* visible
function table type possible (§4.2).

### 4.2 Visible function signature (miss / closesthit / callable)

Exactly one signature, so that every record in one
`visible_function_table` is interchangeable — i.e. real SBT semantics:

```metal
using slang_RTHandler = void(thread slang_RTContext&, device slang_RTGlobals*);

[[visible]] void slang_miss_<mangledName>(thread slang_RTContext& ctx,
                                          device slang_RTGlobals* globals);
```

The generated body unpacks the payload blob into the user's payload type at
entry, runs the user code, and packs it back on exit (the legalizer does the
same load/store sandwich the SPIR-V backend does for `[__vulkanRayPayload]`
globals). Callable data uses the same blob (DXR permits distinct sizes; the
max-size rule applies to both).

### 4.3 Global resources: `slang_RTGlobals`

Visible functions cannot see the kernel's bound resources; every user resource
referenced by *any* linked stage must travel explicitly. Slang aggregates the
union of all stages' global parameters into **one argument buffer struct**
(`slang_RTGlobals`), laid out with the existing Metal argument-buffer /
`ParameterBlock` machinery and exposed through reflection exactly like any
other parameter block. The raygen kernel receives it as an ordinary
`[[buffer(n)]]` parameter (index assigned by normal Metal binding layout) and
threads the pointer through every handler call. Acceleration structures,
textures and samplers inside argument buffers require argument-buffer tier 2
devices — acceptable, since function pointers already restrict us to Apple6+/
Mac2.

### 4.4 The system parameters and the software SBT

The raygen kernel gets these implicit trailing parameters (indices assigned by
normal layout, visible in reflection):

```metal
kernel void <raygenName>(
    /* user + slang_RTGlobals params ... */
    instance_acceleration_structure           slang_rtScene      [[buffer(a)]],
    visible_function_table<slang_RTHandler>   slang_rtHandlers   [[buffer(b)]],
    intersection_function_table<triangle_data, instancing>
                                              slang_rtIsect      [[buffer(c)]], // only if any anyhit/intersection linked
    constant slang_RTSbt&                     slang_rtSbt        [[buffer(d)]],
    uint3 tid  [[thread_position_in_grid]],
    uint3 tdim [[threads_per_grid]])
```

```metal
struct slang_RTSbt
{
    uint missBase;      // first miss record in slang_rtHandlers
    uint hitBase;       // first hit-group (closesthit) record
    uint callableBase;  // first callable record
    uint hitStride;     // records per hit group (== MultiplierForGeometry... convention)
    // per-instance SBT offsets (DXR InstanceContributionToHitGroupIndex);
    // Metal's instance descriptor only carries the *intersection* table offset,
    // so the visible-table contribution is provided here:
    device const uint* instanceSbtOffsets;
};
```

The application fills the tables and this buffer. This is the entire "SBT":
region bases + indices, no GPU handles.

### 4.5 Intersection / anyhit functions

Emitted with Metal's fixed signatures; the payload rides in `ray_data` address
space via `[[payload]]`, which Metal forwards from the `intersect()` call:

```metal
[[intersection(bounding_box, instancing, triangle_data)]]
BoundingBoxResult slang_isect_<name>(
    float3 origin [[origin]], float3 direction [[direction]],
    float minT [[min_distance]], float maxT [[max_distance]],
    uint primitiveIndex [[primitive_id]], uint instanceIndex [[instance_id]],
    ray_data slang_RTContext& ctx [[payload]],
    device slang_RTGlobals* globals /* via intersection-table buffer binding */)
```

- `ReportHit(t, kind, attrs)` → write `attrs`/`kind` into `ctx`, `return
  { true, t }` (bounding-box) / `accept` (triangle anyhit).
- `IgnoreHit()` → `return { false, ... }` / reject.
- `AcceptHitAndEndSearch()` → accept + `intersector::accept_any_intersection`
  semantics (Metal ends search when an intersection function accepts and the
  ray was traced with `accept_any_intersection`, matching the DXR flag).
- anyhit-only groups use Metal `triangle` intersection functions that
  accept/reject the builtin triangle hit.
- Table indexing: Metal's native per-geometry
  `intersectionFunctionTableOffset` + per-instance offset carry DXR's
  hit-group-index semantics for the intersection table; the runtime fills the
  geometry/instance descriptors accordingly.

## 5. Lowering `TraceRay` and `CallShader`

### 5.1 `TraceRay` (hlsl.meta.slang `:19718` gets a `case metal:`)

Lowered (by the legalizer, on IR, not string pasting) to:

```metal
ctx.origin = ray.Origin; ... ctx.rayFlags = RayFlags;
/* pack Payload into ctx.payload */
raytracing::ray r(ray.Origin, ray.Direction, ray.TMin, ray.TMax);
intersector<triangle_data, instancing> i;
/* map RayFlags: force_opacity(), accept_any_intersection(), cull mode; */
/* i.assume_geometry_type(triangle) when no intersection/anyhit stage is linked */
auto hit = i.intersect(r, slang_rtScene, InstanceInclusionMask
                       /* , slang_rtIsect when linked, ctx as [[payload]] */);
if (hit.type != intersection_type::none) {
    /* fill ctx.hitT, instance/geometry/primitive indices, barycentrics */
    uint idx = slang_rtSbt.hitBase
             + RayContributionToHitGroupIndex
             + MultiplierForGeometryContributionToHitGroupIndex * hit.geometry_id
             + slang_rtSbt.instanceSbtOffsets[hit.instance_id];
    slang_rtHandlers[idx](ctx, globals);
} else {
    slang_rtHandlers[slang_rtSbt.missBase + MissShaderIndex](ctx, globals);
}
/* unpack ctx.payload into Payload */
```

### 5.2 `CallShader`

`slang_rtHandlers[slang_rtSbt.callableBase + index](ctx, globals)` with the
callable data through the same blob.

### 5.3 Recursion (`TraceRay` from a closesthit function)

MSL visible functions are ordinary functions; the lowered trace sequence is
legal inside them *if* the system parameters are reachable — which they are,
since `ctx`/`globals` are parameters and the scene/tables can be threaded
through `slang_RTGlobals`. This must be validated on-device early (P3, §8); if
a driver restriction surfaces, the documented fallback is the standard
megakernel iteration (the handler writes a continuation request into `ctx` and
returns; the kernel loops). Either way `DispatchRaysIndex()` etc. keep working
because they read `ctx`, not kernel-only builtins.

## 6. Intrinsics mapping (per-stage availability as in DXR)

| Slang intrinsic | Metal lowering |
|---|---|
| `DispatchRaysIndex/Dimensions` | `ctx.launchIndex` / `ctx.launchDim` (kernel fills from `tid`/`tdim`) |
| `WorldRayOrigin/Direction`, `RayTMin`, `RayFlags` | `ctx` fields |
| `RayTCurrent` | `ctx.hitT` (hit stages), `max_distance` (intersection) |
| `InstanceIndex/InstanceID/GeometryIndex/PrimitiveIndex/HitKind` | `ctx` fields |
| `ObjectRayOrigin/Direction` | intersection stage: `[[origin]]`/`[[direction]]` params; hit stages: computed via instance transform (below) |
| `ObjectToWorld*/WorldToObject*` | from the intersector result (`object_to_world_transform` with the `world_space_data` tag) stored into `ctx`; `WorldToObject` computed by inversion or a runtime-provided per-instance buffer (implementation choice, P4) |
| `ReportHit/IgnoreHit/AcceptHitAndEndSearch` | §4.5 |
| `TraceRay` / `CallShader` | §5 |

## 7. Compiler work plan (file by file)

1. `slang-capabilities.capdef` — §3 alias change (one line) + any
   per-intrinsic tightening.
2. `hlsl.meta.slang` — `case metal:` bodies for the intrinsics; mostly thin
   wrappers over new IR ops (`kIROp_MetalTraceRay`-style) so the real work
   happens in the legalizer, mirroring how the GLSL path uses
   `[__vulkanRayPayload]` globals rather than textual expansion.
3. `slang-ir-metal-legalize.cpp` (`legalizeIRForMetal`, `:408`) — new pass:
   - raygen: rewrite the entry to the kernel form, synthesize `slang_RTContext`,
     append system parameters (§4.4), fill launch state;
   - miss/closesthit/callable: rewrite to the uniform handler signature with
     the payload pack/unpack sandwich;
   - anyhit/intersection: rewrite to Metal intersection-function signatures,
     map `ReportHit`/`IgnoreHit`;
   - lower the trace/call IR ops per §5;
   - hoist stage-global resources into the `slang_RTGlobals` argument buffer
     (reuse the existing parameter-block-to-argument-buffer machinery).
4. `slang-emit-metal.cpp` — stage attribute emission (the `switch` at `:233`):
   `RayGeneration → [[kernel]]`, `Miss/ClosestHit/Callable → [[visible]]`,
   `AnyHit/Intersection → [[intersection(...)]]` with tags derived from what
   the program links (triangle_data, instancing, world_space_data);
   type emission for `visible_function_table<>` /
   `intersection_function_table<>` parameters.
5. Reflection — no new API shape needed: the system parameters and
   `slang_RTGlobals` appear as ordinary (auto-introduced) parameters with
   Metal binding indices; document the naming convention so runtimes can find
   them (`slang_rtScene`, `slang_rtHandlers`, `slang_rtIsect`, `slang_rtSbt`).
6. Tests — `tests/metal/` compile tests per stage + intrinsic; execution
   tests via the standard test harness where Metal execution is available.
   An external end-to-end suite exists (§9) that validates the *same* Slang
   source against Vulkan (lavapipe, software `VK_KHR_ray_tracing_pipeline`)
   and Metal on-device, including an image-level cross-check.

## 8. Phasing

- **P0** — raygen + miss + closesthit, triangle geometry only, `TraceRay`
  from raygen only, payload blob, `slang_RTGlobals`. (Proves the ABI; enough
  for a Whitted-style consumer with shading in raygen.)
- **P1** — anyhit + intersection via the intersection function table
  (`ray_data` payload, `ReportHit` family).
- **P2** — callables.
- **P3** — `TraceRay` from closesthit (recursion depth 2, §5.3) — validate the
  visible-function-intersector question on-device first.
- **P4** — completeness: `ObjectToWorld/WorldToObject`, full ray-flag mapping,
  multi-dimension launches, diagnostics polish (payload/attribute size limits).

## 9. External validation (consumer handoff context)

A bgfx fork (`github.com/natyamatsya/bgfx`, branch `experimental/rt-pipeline`
and `experimental/metal-rt-pipeline-spike`) has:

- a complete RT-pipeline runtime consuming Slang-compiled stages on Vulkan,
  CI-able via Mesa lavapipe, with per-stage smoke tests
  (`tools/rt-validation/rt_pipeline_smoke.cpp`) and a bit-exact ray-query
  vs. RT-pipeline image cross-check on a Cornell Box scene;
- an on-device Metal prototype (`tools/rt-validation/metal_rt_pipeline_spike.cpp`,
  Apple M2 Max) of exactly the §2 mapping — visible-function SBT routing and a
  bounding-box intersection function both verified (4096/4096 pixels, analytic
  distance validated), i.e. **the runtime side of this spec is already proven**;
  design notes in that repo's `METAL_RT_PIPELINE.md`.

The moment P0 lands, that consumer exercises it end-to-end on Metal with zero
new test code (its Metal backend implements the §4 contract, and its existing
lavapipe-vs-Metal comparison validates images across targets).

## 10. P0 implementation notes (what landed, and where it deviates)

P0 landed in:

- `source/slang/slang-capabilities.capdef` — `raytracing` alias gained a
  `metallib_2_4` arm; the P0 intrinsics' `[require(...)]` target lists gained
  `metal`.
- `source/slang/hlsl.meta.slang` — `case metal:` bodies for `TraceRay` and
  the system-value intrinsics, plus the internal `__metalRTCall*Handler` /
  `__metalRT*` intrinsic-op declarations.
- `source/slang/slang-ir-insts.lua` — the transient `metalRT*` ops, the
  emit-time `metalRTHandlerCall` op, and `MetalVisibleFunctionTableType`.
- `source/slang/slang-ir-metal-legalize-raytracing.cpp` — the new legalizer
  pass (`legalizeMetalRayTracing`), run first in `legalizeIRForMetal`.
- `source/slang/slang-ir-explicit-global-context.cpp` — handler-stage entry
  points are excluded from kernel-context parameter threading on Metal.
- `source/slang/slang-type-layout.cpp` — Metal layout rules for the ray
  payload / callable payload / hit attribute resource kinds (previously
  null, which crashed entry-point layout).
- `source/slang/slang-emit-metal.cpp` — `[[kernel]]` / `[[visible]]` stage
  attributes, `visible_function_table<...>` type emission, and the
  `table[index](ctx, globals)` call form.
- Tests: `tests/metal/raytracing-pipeline*.slang`. The generated MSL for the
  P0 pipeline compiles cleanly with the Apple `metal` compiler (`-std=metal3.1`).

Deviations from the text above — the runtime-visible ones first:

1. **System parameter bindings are fixed, not layout-assigned** (§4.4): the
   raygen kernel's implicit parameters are bound at the top of Metal's
   `[[buffer]]` index range, where normally-laid-out user parameters
   (assigned from 0 upward) cannot collide:
   `slang_rtGlobals` = `[[buffer(28)]]`, `slang_rtSbt` = `[[buffer(29)]]`,
   `slang_rtHandlers` = `[[buffer(30)]]`. They are consequently not part of
   reflection yet; layout-assigned + reflected bindings can replace this
   later without changing the handler ABI.
2. **No `slang_rtScene` system parameter** (§4.4): in Slang the acceleration
   structure is the user's own global parameter (`TraceRay`'s first
   argument), so it is bound like any other resource, exactly as the
   already-shipping `RayQuery` path binds it. A separate implicit scene
   binding would be a second representation of the same value.
3. **Handler parameters are pointers, not references** (§4.2): the uniform
   signature is `void(slang_RTContext thread*, slang_RTGlobals device*)`.
   Both the call site and the `[[visible]]` definitions are emitted by
   Slang from the same IR types, so the choice is self-consistent, and MSL
   lowers references to pointers at the AIR level anyway.
4. **Handlers keep their user-facing entry-point names** (§4.2 showed
   `slang_miss_<mangledName>`): the application looks functions up by the
   entry-point name it already knows; no prefix scheme is applied.
5. **Traversal reuses the `intersection_query` lowering, not `intersector<>`**
   (§5.1): for the P0 scope (triangle geometry, no intersection/anyhit
   functions linked) the two Metal APIs drive the same hardware traversal
   with identical semantics, and the query form is what the Metal backend
   already emits and tests for `RayQuery`. Non-opaque triangle candidates
   are committed unconditionally, which is DXR's default-accept anyhit
   behavior when no anyhit shader exists. P1 (intersection function tables)
   revisits this, since only `intersector<>::intersect()` accepts a
   `MTLIntersectionFunctionTable`.
6. **`slang_RTGlobals` hoisting is not implemented yet** (§4.3): the struct
   exists (with one reserved `uint` field, since MSL has no empty structs)
   and is threaded through the whole ABI, but globally bound resources are
   not yet moved into it; a miss/closest-hit shader that references a
   global parameter — directly or transitively through helper functions it
   calls — is diagnosed (`metal-raytracing-global-param-in-handler`,
   E56113) instead of silently miscompiling. As a backstop for the same
   invariant, the kernel-context pass
   (`slang-ir-explicit-global-context.cpp`) also diagnoses if it would ever
   have to append a context parameter to a handler entry point, since that
   would corrupt the fixed visible-function signature the kernel calls
   through. Ray-generation shaders keep full access to ordinary bindings.
7. **Payload size is a fixed 64 bytes** (§4.1): the
   `-metal-rt-max-payload-size` option is not implemented yet; exceeding the
   blob is a compile-time error (E56112), as specified.
8. **`RayTCurrent()` in a miss shader** reads `ctx.hitT`, which the miss
   dispatch fills with the ray's `TMax` — one context field serves both the
   closest-hit and miss semantics of DXR.
9. **Unsupported phases fail loudly**: `anyhit` / `intersection` /
   `callable` entry points (E56110), `TraceRay` outside a raygeneration
   entry point (E56111), unsupported intersection attribute types (E56114),
   and ray-tracing intrinsics that survive outside any supported entry
   point (E56115) are compile-time errors until P1–P3 land.
10. **Reserved binding collisions are diagnosed** (follows from note 1):
   a user resource laid out or explicitly bound at `buffer(28)` or above
   in a program with a raygeneration entry point is rejected (E56116)
   rather than silently double-bound against the system parameters.

### P1 implementation notes (anyhit + intersection)

P1 replaced the P0 traversal and added the intersection-function stages:

- **Unified `intersector<>` traversal** (supersedes deviation 5 above): every
  `TraceRay` now lowers through one transient IR op whose traversal half is
  the `_slang_rtTrace` prelude template (`slang-emit-metal-prelude.cpp`): it
  configures an `intersector<triangle_data, instancing>` from the DXR ray
  flags, runs `intersect()` — with the `slang_rtIsect` intersection function
  table when the program links anyhit/intersection stages, without it
  otherwise — and fills the committed-hit context fields. The SBT dispatch
  half stays in IR (`slang-ir-metal-legalize-raytracing.cpp`).
- **`slang_rtIsect` at `[[buffer(27)]]`**, bound only when anyhit or
  intersection entry points are present *in the same compiled module* as the
  ray-generation kernel. A raygen module compiled separately from its
  intersection stages would not receive the table binding — compile the
  stages together until a forcing option exists. The reserved-binding
  collision guard's floor moves to 27 accordingly.
- **Anyhit** becomes `[[intersection(triangle, ...)]] bool f(...)` with the
  candidate state as tagged parameters (`[[distance]]`,
  `[[barycentric_coord]]`, `[[front_facing]]`, ids) and the context as the
  `ray_data ... & [[payload]]` reference. Falling off the end accepts;
  `IgnoreHit()` returns false; `AcceptHitAndEndSearch()` accepts (the search
  actually ends only when the ray carries
  `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`, per §4.5). The payload is
  accessed directly through a typed view of the context blob so writes
  persist for ignored hits, as DXR requires. `HitKind()` derives from
  `[[front_facing]]`.
- **Intersection** becomes `[[intersection(bounding_box, ...)]]` returning
  `slang_RTBBoxResult { bool accept [[accept_intersection]]; float distance
  [[distance]]; }`. `ReportHit(t, kind, attrs)` is range-guarded: when `t`
  lies inside `[[min_distance]], [[max_distance]]`, the attributes are
  stored into the context's 32-byte attribute blob, the hit kind into the
  context, and the function returns the accepting result — code after an
  in-range `ReportHit` therefore does not run (single-report semantics; DXR
  shaders that keep reporting after an accepted hit are not expressible on
  Metal's return-once protocol). An out-of-range `ReportHit` yields `false`
  and execution continues. Writing only in-range reports keeps the blob
  consistent: Metal commits every accepted in-range hit, so the last write
  is always the final committed hit. Hit groups combining an intersection
  shader *and* an anyhit shader are not supported (Metal has no anyhit
  chaining after a bounding-box report; fuse the logic into the
  intersection shader).
- **Closest-hit attribute rule**: the built-in single-`float2` triangle
  attribute struct reads the committed barycentrics field; any other
  attribute struct reads the attribute blob (E56114 if the attribute type
  is not a struct, E56117 if it exceeds 32 bytes).
- **Pinned ABI names**: `slang_RTContext` / `slang_RTGlobals` / `slang_RTSbt`
  / `slang_RTBBoxResult` and all their field names emit without uniqueness
  suffixes, so separately compiled stage libraries produce byte-identical
  struct declarations (and the prelude template can address context fields
  by name).
- E56110 now rejects only `callable` (phase P2).

### P2 implementation notes (callables)

- A callable shader is, from the execution model's point of view, a miss
  shader with user-chosen data: it is rewritten to the same uniform
  `[[visible]]` handler signature, its `inout` data parameter bridged
  through the context's payload blob, and it occupies records in the same
  visible function table (from the `callableBase` region of `slang_RTSbt`).
- `CallShader(index, data)` lowers (in the ray-generation kernel) to: pack
  `data` into the payload blob, call record `callableBase + index` through
  `slang_rtHandlers`, unpack the blob back. The 64-byte payload size limit
  applies to callable data as well (§4.2).
- `CallShader` from a miss/closest-hit/callable shader is diagnosed
  (E56111) for the same reason as `TraceRay` recursion: handler stages do
  not receive the table/SBT system parameters until the phase-P3 mechanism
  (threading them through `slang_RTGlobals`) exists. Note that a
  `CallShader` between two `TraceRay`s in raygen reuses the same payload
  blob, which is safe because each dispatch packs before and unpacks after
  the call.
- With all six stages supported, the `metal-raytracing-stage-not-supported`
  diagnostic (E56110) was retired.

## 11. Open questions

1. `intersector<>` inside `[[visible]]` functions (P3): believed legal (they
   are ordinary AIR functions), needs an on-device proof before committing to
   recursion; megakernel fallback specified.
2. Payload blob size: fixed max (this spec) vs. per-payload-type table
   specialization (rejected for now: breaks single-table SBT semantics and
   explodes pipeline variants).
3. `WorldToObject` source: intersector `world_space_data` tags vs. a
   runtime-supplied per-instance transform buffer (P4 decision).
4. Argument-buffer tier constraints for acceleration structures inside
   `slang_RTGlobals` on older macOS versions — may need a minimum-OS note in
   the user guide.
