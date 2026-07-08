# Metal Ray-Tracing Runtime Contract v2 — Separate Compilation and the Globals ABI

Status: **reviewed, revision 2** (co-design spec; companion to `metal-raytracing.md`,
whose compiler half is implemented through pay-for-use `world_space_data`; revision 2
resolves the review's blocking finding -- the implicit uniform constant buffer under
slots mode -- and its text-level items)
Scope: the three deliberately-open runtime-contract items — hoisted-globals layout
pinning, reflection-visible system bindings, anyhit/intersection globals.

## 1. The problem, named once

Every open item is the same problem wearing different clothes: **the compiler makes
per-module decisions that are part of a cross-module ABI.** A Metal ray-tracing
pipeline is assembled by the application from separately compiled stage functions, so
any layout or tag set the compiler derives from *what this module happens to use*
breaks the moment the consuming runtime compiles one stage per binary — which is
exactly what bgfx's shaderc does (one envelope per stage), and what any engine with
per-shader caching will do.

The existing force options (`-metal-rt-force-isect-table`,
`-metal-rt-force-world-space-data`, `-metal-rt-max-payload-size`) already fix three
instances of this by making the decision explicit and uniform. This spec finishes the
job for the remaining implicit decisions, and adds the piece that makes all of them
*checkable*: a machine-readable ABI descriptor per compiled module.

The consuming-runtime facts this design must serve (from the working bgfx
integration):

- Stages compile independently; no compile-time channel exists between them other
  than command-line flags chosen by the build system.
- The runtime today discovers the uniform-buffer slot via Metal pipeline reflection,
  and the hoisted-globals tail by parsing the emitted `struct slang_RTGlobals` text —
  a working but informal contract.
- The current hoisting rule ("tail = the resources this module's handlers reference,
  in IR-global order") only composes across modules in the degenerate case of a
  single resource-using handler.

## 2. The ABI descriptor (makes the contract checkable)

Every compiled RT-stage module gets a single machine-readable comment line at the top
of the emitted MSL:

```
// slang-metal-rt-abi:{"v":1,"payload":64,"attr":32,"ws":0,"isect":1,"slots":8,"sys":{"handlers":30,"sbt":29,"globals":28,"isect":27},"uniforms":{"buf":1,"size":32}}
```

Format: **strictly one physical line** (it fits; no continuation syntax exists, so a
parser never guesses). Fields in fixed order; the scalar fields are always present;
two **optional trailing objects** with explicit presence rules:

- `sys` — present in ray-generation modules only (the kernel owns the system
  bindings).
- `uniforms` — present in any module that references loose uniforms: in a raygen
  module `buf` is the implicit constant buffer's Metal buffer index; in a handler
  module `buf` is absent and `size` states the scratch-region size the module's
  hoisted `uniforms` header entry (§3) will read — the runtime needs it either way.

| field | meaning |
|---|---|
| `v` | descriptor version |
| `payload` / `attr` | payload and attribute blob sizes (bytes) |
| `ws` | `world_space_data` on (1) / off (0) |
| `isect` | intersection-function-table support on/off |
| `slots` | globals layout mode: `0` = usage-derived (§3 legacy), `N>0` = slot-addressed with N slots |
| `sys` | the reserved system buffer indices of this module's kernel (raygen modules only) |
| `uniforms` | the implicit constant buffer's byte size, plus its Metal buffer index in raygen modules (see presence rules above) |

Rationale and obligations:

- **Emission is cheap and total** — with the ownership plumbed correctly: the
  payload/`ws`/`slots` values are the *legalizer's* decisions, so the legalizer
  records them as a module decoration and the emitter prints the record
  (decision-owner writes it). The decoration doubles as the future transport for a
  reflection-API query (§8 Q1) and for any binary target. No new file formats — the
  descriptor rides in the artifact it describes and survives any envelope/packaging
  scheme that carries the MSL (bgfx envelopes do).
- **The runtime's duty**: at program creation, parse the descriptor from every stage
  module and **verify agreement** of the cross-module fields (`payload`, `attr`,
  `ws`, `isect`, `slots`). Mismatch = a clear creation-time error naming the two
  disagreeing stages — replacing today's failure mode (GPU misbehavior with no
  diagnostic). This is the single biggest robustness win of the spec.
- **It formalizes what already works**: the bgfx runtime stops scraping struct
  declarations and reads one versioned line. The Metal-pipeline-reflection path for
  the uniform slot remains valid; `uniforms.buf` makes it redundant.
- **End state**: when this stack upstreams, the descriptor's content graduates into
  the Slang reflection API proper (an RT-ABI query on the entry point). The comment
  line is v1 of the *contract*, not necessarily the final *transport*; the field set
  is the durable part.

## 3. Slot-addressed globals layout (`-metal-rt-globals-slots N`)

The tail of `slang_RTGlobals` becomes **positionally addressed by the resource's
declared `register()` number** instead of derived from usage:

```
-metal-rt-globals-slots 8
```

- The struct is emitted with the fixed dispatch header — `handlers`, `isect`, `sbt`,
  **and `uniforms`** — followed by **exactly N 8-byte slots**. Slot `k` belongs to the
  resource declared with register number `k` (`tN`/`uN` share one index space, as the
  consuming-runtime convention already requires globally unique register numbers).
- **The `uniforms` header entry (resolves the review's blocking finding)**: the
  implicit global constant buffer (`u_params` and friends) cannot carry a
  `register()`, yet handler stages read it through the globals buffer — the most
  common consuming shape. Under slots mode the header therefore always ends with
  `constant GlobalParams* uniforms;` (8 bytes; the runtime encodes the dispatch's
  uniform-scratch device address, or 0 when the program has no loose uniforms).
  Slots start at byte 48. This keeps by-construction layout agreement and costs one
  pointer; the descriptor's `uniforms.size` (§2) tells the runtime how large the
  scratch region handlers will read. The header layout, normatively (and pinned
  executably by the mixed-module kit, §7):

  | offset | size | field |
  |---|---|---|
  | 0  | 8  | `handlers` (visible function table, `gpuResourceID`) |
  | 8  | 8  | `isect` (intersection function table, `gpuResourceID`; 0 = none) |
  | 16 | 24 | `sbt` (inline; the on-device-validated `slang_RTSbt` layout) |
  | 40 | 8  | `uniforms` (`constant GlobalParams*`; 0 = no loose uniforms) |
  | 48 | 8·N | slots 0..N-1 |
- A module emits a **typed field** for every register its handler stages reference
  (`device T*`, `acceleration_structure<...>`, `texture2d<...>` — all 8 bytes in an
  argument buffer) and an **opaque placeholder** (`uint64_t slang_rtSlot<k>;`) for
  every other slot. Layouts agree across modules *by construction*; types are each
  module's local view of the same slot.
- `N` is a cross-module ABI field (in the descriptor, verified by the runtime).
  `slots:0` keeps today's usage-derived layout for whole-pipeline single-module
  compiles (tests, tools) — nothing existing changes behavior.
- **Runtime encoding becomes trivial**: slot `k` is filled from the application
  binding at stage `k` (buffer → `gpuAddress`, AS/texture → `gpuResourceID`). The
  per-handler "layout owner" concept and the single-resource-using-handler
  restriction disappear.
- Resources without an explicit `register()` are a compile-time error under slots
  mode (the register *is* the slot address; the consuming convention already
  requires it). Likewise diagnosed: **duplicate slot addresses** (`t2` and `u2`
  collide in the shared index space) and **register spaces** (`space1` etc. are
  unsupported in v1).
- **Encoding floor**: the 8-byte-slot contract assumes the raw resource-ID encoding
  the validation kits proved — `gpuAddress`/`gpuResourceID` written at fixed offsets,
  i.e. Metal-3-era argument buffers. The MSL itself compiles at `macos-metal2.4`, but
  a 2.4-era `MTLArgumentEncoder`-layout runtime is out of contract. **Samplers are
  excluded from v1 slots** (untested in the kits; revisit with the texture-slot
  question, §8 Q3).

Why not alternatives considered:

- *Declaration-driven ordering* (tail = declared resources in order) still differs
  across modules that declare different subsets — it moves the problem, positional
  addressing removes it.
- *A two-pass layout manifest* (first compile emits, others consume) couples the
  build graph and fails caching; rejected for toolchain complexity.

## 4. Anyhit/intersection globals via the intersection-table buffer path

Metal intersection functions may declare trailing buffer parameters, bound through
`MTLIntersectionFunctionTable::setBuffer(buffer, offset, index)`. That is the
missing globals channel for the two stages whose fixed signatures have no globals
pointer today:

- **Compiler**: when an anyhit/intersection entry point references a globally bound
  resource, the function gains one trailing parameter —
  `device slang_RTGlobals* slang_rtGlobals [[buffer(0)]]` — and global
  references rewrite to loads from it, exactly as miss/closest-hit handlers do
  today. E56113 narrows again: only module-scope *mutable state* remains rejected.
- **Runtime**: bind the (one, shared) globals buffer at **intersection-table buffer
  slot 0** whenever the program has an intersection table. IFT buffer slot 0 is
  hereby reserved by the contract.
- **No new cross-module switch is needed**: functions that do not declare the
  trailing parameter simply do not read the binding; binding slot 0 unconditionally
  is safe. (The descriptor still lets the runtime *know* whether any stage reads it,
  for residency bookkeeping.)
- Semantics note: anyhit/intersection run *during* traversal. Reading globals is
  well-defined; dispatching through the handler table remains out of contract for
  these stages (unchanged — they carry no table state).

## 5. Flag discipline (the build system's side of the bargain)

For a multi-module pipeline the build passes an identical ABI flag set to every
stage compile:

```
-metal-rt-max-payload-size <S> -metal-rt-globals-slots <N>
[-metal-rt-force-isect-table] [-metal-rt-force-world-space-data]
```

- The force options are needed exactly when *some other* module makes the feature
  real (an intersection stage exists; some stage uses transforms). A build system
  that knows the program shape sets them per program; one that does not may set
  them unconditionally at a known, bounded cost (documented: the isect table is a
  binding, `world_space_data` is ~128 bytes of per-thread context plus the tag).
- The descriptor (§2) is the enforcement: the runtime rejects mixed flag sets at
  program creation, so a build-system mistake is a readable error, not corruption.

## 6. Work plan

Implementation status: **C1, C2, and C3 are implemented on `metal-rt-impl`**
(the descriptor line, `-metal-rt-globals-slots`, and the intersection-table
globals path — E56113 is narrowed to module-scope mutable state). One
consuming-side note from C2: Slang's pre-existing warning 39029 (D3D
register without a Vulkan binding) fires on `register()` declarations in
Metal-only compiles — stage compiles should pass `-warnings-disable 39029`.

Compiler (slang, `metal-rt-impl`):

1. **C1 — descriptor**: the legalizer records its ABI decisions as a module
   decoration; the emitter prints the one-line descriptor from it
   (`slang-ir-metal-legalize-raytracing.cpp` + `slang-emit-metal.cpp`). Tests pin
   the line's shape for a lean and an opted-in module.
2. **C2 — slots mode**: `-metal-rt-globals-slots` option; `hoistHandlerGlobals`
   gains the positional path (register from the declaration, placeholder fields for
   unused slots, the `uniforms` header entry, diagnostics for missing `register()`
   / out-of-range / duplicate slot / register spaces); descriptor reports `slots`.
3. **C3 — intersection-stage globals**: widen `hoistHandlerGlobals`' collection set
   to anyhit/intersection entry points (so the tail is complete before layout
   assignment); trailing `[[buffer(0)]]` parameter on functions that need it; reuse
   `rewriteHoistedGlobalUses`; narrow E56113.

Runtime (bgfx, `experimental/metal-rt-pipeline`):

1. **R1** (with C1): parse + cross-verify descriptors in `createRtProgram`; drop the
   `slang_RTGlobals` text scan and the reflection-based uniform-slot discovery in
   favor of descriptor fields.
2. **R2** (with C2): shaderc passes the uniform flag set (`--metal-rt-*` plumbed
   through); the globals encoder becomes "slot k ← bgfx stage k" plus the `uniforms`
   header entry filled with the dispatch's uniform-scratch device address; the
   single-resource-using-handler restriction and the layout-owner logic are deleted.
3. **R3** (with C3): `ift->setBuffer(globalsBuf, 0, 0)` when an IFT exists; extend
   residency to tail resources referenced by intersection stages.

## 7. Validation plan (extends the existing kit series)

- **The mixed-module kit (the point of the whole spec)**: stages compiled as
  *separate* slangc invocations with the uniform flag set — raygen alone, two
  handlers with *different* resource subsets, an intersection stage reading a
  global — assembled by a host that encodes the slot table once. One handler reads
  loose uniforms (`u_params`) compiled separately from the raygen, pinning the
  `uniforms` header entry. This deliberately breaks the "Cornell shape" and is the
  first on-device proof of the separate-compilation ABI. Failure signatures per
  slot, additive values as always.
- **Descriptor-mismatch negative test**: one stage compiled with a different payload
  size → program creation must fail with the named-stages error (runtime-side test).
- **Cornell under slots mode**: the stage-5 referee re-run with shaderc passing
  `-metal-rt-globals-slots`, proving the production path end-to-end; lavapipe
  unaffected (all options are Metal-target no-ops elsewhere — worth one compile
  assertion).

## 8. Open questions

1. Descriptor transport for non-text targets: if a metallib (binary) target arrives,
   the comment line needs a sibling (a `[[section]]`-style blob or the reflection
   API arriving early).
2. ~~Slot count default~~ **Decided (review)**: v1 requires an explicit N — an
   implicit default is a silent ABI choice, the exact thing this spec exists to
   eliminate. Revisit ergonomics after soak.
3. Texture slots: `gpuResourceID` encoding for textures in the tail is specified but
   untested on-device in the current kits (buffers and acceleration structures are);
   the mixed-module kit should include one texture slot.
4. ~~Default mode~~ **Decided (review)**: usage-derived stays the default until the
   mixed-module kit has soaked; flipping the default is a one-line change later.
