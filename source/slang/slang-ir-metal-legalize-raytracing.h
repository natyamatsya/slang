// slang-ir-metal-legalize-raytracing.h
#pragma once

#include "slang-ir-legalize-varying-params.h"

namespace Slang
{

struct IRModule;
class DiagnosticSink;
class TargetProgram;

/// Is `stage` one of the ray-tracing stages that the Metal target emits as a
/// function invoked through a function table rather than as a kernel?
///
/// Such functions have a fixed signature — for miss/closest-hit/callable the
/// uniform handler ABI `void(slang_RTContext thread*, slang_RTGlobals
/// device*)` of docs/design/metal-raytracing.md section 4.2 — so no pass may
/// add parameters (e.g. resource bindings or a kernel context) to them. The
/// ray-generation stage is not in this set: it becomes the compute kernel and
/// keeps ordinary bindings.
bool isMetalRayTracingHandlerStage(Stage stage);

/// Rewrite ray-tracing pipeline entry points and the transient `kIROp_MetalRT*`
/// instructions into the Metal execution model specified by
/// docs/design/metal-raytracing.md.
///
/// Ray-generation entry points become ordinary compute kernels that own a local
/// `slang_RTContext` and receive the `slang_rtHandlers` visible function table,
/// the `slang_rtSbt` software shader-binding-table buffer, and the
/// `slang_rtGlobals` argument buffer as trailing parameters. Miss and
/// closest-hit entry points are rewritten to the uniform handler signature
/// `void(slang_RTContext thread*, slang_RTGlobals device*)`, with their payload
/// (and hit-attribute) parameters bridged through the context's payload blob.
/// The `metalRTCall{Miss,Hit}Handler` ops produced by `TraceRay` lower to the
/// DXR shader-binding-table indexing computation plus a `metalRTHandlerCall`
/// through the visible function table, and the `metalRT*` system-value reader
/// ops lower to loads of the corresponding context fields.
void legalizeMetalRayTracing(
    IRModule* module,
    TargetProgram* targetProgram,
    DiagnosticSink* sink,
    List<EntryPointInfo>& entryPoints);

} // namespace Slang
