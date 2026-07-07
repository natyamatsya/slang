// slang-ir-metal-legalize-raytracing.cpp
#include "slang-ir-metal-legalize-raytracing.h"

#include "slang-ir-inline.h"
#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir-util.h"
#include "slang-ir.h"
#include "slang-rich-diagnostics.h"

namespace Slang
{

// The Metal ray-tracing ABI of docs/design/metal-raytracing.md: cross-stage
// payload and attribute data travel through fixed-size byte blobs inside the
// per-thread `slang_RTContext`, so that a single visible-function-table
// signature can serve every miss/closest-hit shader (real shader-binding-table
// semantics). The sizes match DXR driver conventions; exceeding the payload
// blob is a compile-time error (`metal-raytracing-payload-too-large`).
static const IRIntegerValue kMetalRTMaxPayloadSize = 64;
static const IRIntegerValue kMetalRTMaxAttributeSize = 32;

// Fixed Metal buffer binding indices for the implicit system parameters that
// a ray-generation kernel receives. These sit at the top of Metal's [[buffer]]
// index range (max 30) so they cannot collide with normally laid-out user
// parameters, which are assigned from 0 upward. The runtime contract is
// documented in docs/design/metal-raytracing.md section 4.4.
static const int kMetalRTGlobalsBufferIndex = 28;
static const int kMetalRTSbtBufferIndex = 29;
static const int kMetalRTHandlersBufferIndex = 30;

// DXR hit-kind values for fixed-function triangle hits.
static const int kHitKindTriangleFrontFace = 254;
static const int kHitKindTriangleBackFace = 255;

// The DXR system-value reader ops and the `slang_RTContext` field each one
// loads. This table is the single source of truth relating the two: it
// expands into both the op classification (`isMetalRTOp`) and the lowering
// (`getContextKeyForReaderOp`).
//
// Note the double duty of `hitT`: `RayTCurrent()` reads the committed-hit
// distance in a closest-hit shader and the ray's tMax in a miss shader; the
// trace lowering stores tMax into `hitT` on the miss path so one field
// serves both (matching DXR, where RayTCurrent in a miss shader equals
// RayTMax).
//
// clang-format off
#define SLANG_METAL_RT_SYSTEM_VALUE_OPS(M)          \
    M(MetalRTDispatchRaysIndex,      launchIndexKey)    \
    M(MetalRTDispatchRaysDimensions, launchDimKey)      \
    M(MetalRTWorldRayOrigin,         originKey)         \
    M(MetalRTWorldRayDirection,      directionKey)      \
    M(MetalRTRayTMin,                tMinKey)           \
    M(MetalRTRayTCurrent,            hitTKey)           \
    M(MetalRTRayFlags,               rayFlagsKey)       \
    M(MetalRTInstanceIndex,          instanceIndexKey)  \
    M(MetalRTInstanceID,             instanceIDKey)     \
    M(MetalRTGeometryIndex,          geometryIndexKey)  \
    M(MetalRTPrimitiveIndex,         primitiveIndexKey) \
    M(MetalRTHitKind,                hitKindKey)
// clang-format on

namespace
{

struct MetalRayTracingLegalizationContext
{
    IRModule* module;
    TargetProgram* targetProgram;
    DiagnosticSink* sink;

    // The shared ABI types, created on first use by `ensureSharedTypes`.
    IRStructType* ctxStructType = nullptr;
    IRFuncType* handlerFuncType = nullptr;
    IRType* ctxPtrType = nullptr;
    IRType* globalsPtrType = nullptr;
    IRType* sbtPtrType = nullptr;
    IRType* visibleTableType = nullptr;

    // Field keys of `slang_RTContext` that the lowering reads or writes.
    // (The struct has more fields than this — e.g. the attribute blob that
    // only intersection-function support will use — but keys are only kept
    // for the fields this pass touches.)
    IRStructKey* originKey = nullptr;
    IRStructKey* tMinKey = nullptr;
    IRStructKey* directionKey = nullptr;
    IRStructKey* tMaxKey = nullptr;
    IRStructKey* rayFlagsKey = nullptr;
    IRStructKey* hitTKey = nullptr;
    IRStructKey* instanceIndexKey = nullptr;
    IRStructKey* instanceIDKey = nullptr;
    IRStructKey* geometryIndexKey = nullptr;
    IRStructKey* primitiveIndexKey = nullptr;
    IRStructKey* hitKindKey = nullptr;
    IRStructKey* triBarycentricsKey = nullptr;
    IRStructKey* payloadKey = nullptr;
    IRStructKey* launchIndexKey = nullptr;
    IRStructKey* launchDimKey = nullptr;

    // Field keys of `slang_RTSbt` that the trace lowering reads.
    IRStructKey* missBaseKey = nullptr;
    IRStructKey* hitBaseKey = nullptr;
    IRStructKey* instanceSbtOffsetsKey = nullptr;

    // Per-entry-point access paths to the ABI values: the context pointer is
    // a local variable in a ray-generation kernel and the first parameter of
    // a miss/closest-hit handler; the table/SBT pointers only exist in the
    // kernel, which is the only place a trace may be lowered.
    struct RTEntryInfo
    {
        Stage stage = Stage::Unknown;
        IRInst* ctxPtr = nullptr;
        IRInst* globalsPtr = nullptr;
        IRInst* handlerTable = nullptr;
        IRInst* sbtPtr = nullptr;
    };
    Dictionary<IRFunc*, RTEntryInfo> rtEntries;

    // Payload types already reported as exceeding the payload blob, so one
    // oversized type diagnoses once even though it is bridged at several
    // points (trace hit path, trace miss path, handler entry/exit).
    HashSet<IRType*> diagnosedOversizePayloadTypes;
};

} // anonymous namespace

bool isMetalRayTracingHandlerStage(Stage stage)
{
    switch (stage)
    {
    case Stage::Miss:
    case Stage::ClosestHit:
    case Stage::AnyHit:
    case Stage::Intersection:
    case Stage::Callable:
        return true;
    default:
        return false;
    }
}

static bool isMetalRTOp(IRInst* inst)
{
    switch (inst->getOp())
    {
#define CASE(OP, KEY) case kIROp_##OP:
        SLANG_METAL_RT_SYSTEM_VALUE_OPS(CASE)
#undef CASE
    case kIROp_MetalRTCallMissHandler:
    case kIROp_MetalRTCallHitHandler:
        return true;
    default:
        return false;
    }
}

/// Map a `metalRT*` system-value reader op to the `slang_RTContext` field it
/// reads. Returns null for ops that are not plain readers.
static IRStructKey* getContextKeyForReaderOp(MetalRayTracingLegalizationContext& context, IROp op)
{
    switch (op)
    {
#define CASE(OP, KEY) \
    case kIROp_##OP:  \
        return context.KEY;
        SLANG_METAL_RT_SYSTEM_VALUE_OPS(CASE)
#undef CASE
    default:
        return nullptr;
    }
}

static bool funcContainsMetalRTOp(IRFunc* func)
{
    for (auto block : func->getBlocks())
        for (auto inst : block->getChildren())
            if (isMetalRTOp(inst))
                return true;
    return false;
}

/// Inline every call to a non-entry-point function that (transitively)
/// contains a Metal ray-tracing op, so that all such ops end up physically
/// inside their entry points, where the context variable and the system
/// parameters are in scope. `TraceRay` itself is `[ForceInline]`, so its body
/// is already inlined into the user's calling function; this handles user
/// helper functions that call `TraceRay` or read ray-tracing system values.
static void inlineMetalRTOpsIntoEntryPoints(IRModule* module)
{
    // Each round inlines every current call site of every RT-op-containing
    // function; inlining can surface new RT-op-containing callers, so repeat
    // until a fixed point. The call graph is acyclic (Slang has no recursion),
    // so this terminates.
    for (;;)
    {
        bool changed = false;
        List<IRCall*> callsToInline;
        for (auto inst : module->getGlobalInsts())
        {
            auto func = as<IRFunc>(inst);
            if (!func)
                continue;
            if (func->findDecoration<IREntryPointDecoration>())
                continue;
            if (!funcContainsMetalRTOp(func))
                continue;
            for (auto use = func->firstUse; use; use = use->nextUse)
            {
                auto call = as<IRCall>(use->getUser());
                if (call && call->getCallee() == func)
                    callsToInline.add(call);
            }
        }
        for (auto call : callsToInline)
        {
            if (inlineCall(call))
                changed = true;
        }
        if (!changed)
            return;
    }
}

static void ensureSharedTypes(MetalRayTracingLegalizationContext& context)
{
    if (context.ctxStructType)
        return;

    IRBuilder builder(context.module);
    builder.setInsertInto(context.module->getModuleInst());

    auto uintType = builder.getUIntType();
    auto floatType = builder.getBasicType(BaseType::Float);
    auto float2Type = builder.getVectorType(floatType, 2);
    auto float3Type = builder.getVectorType(floatType, 3);
    auto uint3Type = builder.getVectorType(uintType, 3);
    auto byteType = builder.getBasicType(BaseType::UInt8);

    auto addField = [&](IRStructType* structType, const char* name, IRType* fieldType)
    {
        auto key = builder.createStructKey();
        builder.addNameHintDecoration(key, UnownedStringSlice(name));
        builder.createStructField(structType, key, fieldType);
        return key;
    };

    // `slang_RTContext` (docs/design/metal-raytracing.md section 4.1): the
    // single thread-local struct through which all cross-stage communication
    // happens. The field order is part of the ABI; the kernel and the visible
    // functions are all emitted from this one type.
    auto ctxType = builder.createStructType();
    builder.addNameHintDecoration(ctxType, UnownedStringSlice("slang_RTContext"));
    context.ctxStructType = ctxType;

    context.originKey = addField(ctxType, "origin", float3Type);
    context.tMinKey = addField(ctxType, "tMin", floatType);
    context.directionKey = addField(ctxType, "direction", float3Type);
    context.tMaxKey = addField(ctxType, "tMax", floatType);
    context.rayFlagsKey = addField(ctxType, "rayFlags", uintType);
    context.hitTKey = addField(ctxType, "hitT", floatType);
    context.instanceIndexKey = addField(ctxType, "instanceIndex", uintType);
    context.instanceIDKey = addField(ctxType, "instanceID", uintType);
    context.geometryIndexKey = addField(ctxType, "geometryIndex", uintType);
    context.primitiveIndexKey = addField(ctxType, "primitiveIndex", uintType);
    context.hitKindKey = addField(ctxType, "hitKind", uintType);
    context.triBarycentricsKey = addField(ctxType, "triBarycentrics", float2Type);
    // The attribute blob is only written once intersection-function support
    // (phase P1) lands, but it is part of the context layout from day one so
    // that P1 does not change the ABI.
    addField(
        ctxType,
        "attributes",
        builder.getArrayType(
            byteType,
            builder.getIntValue(builder.getIntType(), kMetalRTMaxAttributeSize)));
    context.payloadKey = addField(
        ctxType,
        "payload",
        builder.getArrayType(
            byteType,
            builder.getIntValue(builder.getIntType(), kMetalRTMaxPayloadSize)));
    context.launchIndexKey = addField(ctxType, "launchIndex", uint3Type);
    context.launchDimKey = addField(ctxType, "launchDim", uint3Type);

    // `slang_RTGlobals` (section 4.3): the argument-buffer struct that carries
    // globally bound resources into the visible functions, which cannot see
    // the kernel's bindings. Hoisting user globals into it is not implemented
    // yet, but the struct is part of the handler signature from day one so
    // the ABI does not change when hoisting lands. It carries one reserved
    // field until then: MSL has no empty structs (an empty IR struct would be
    // folded to `void`, and `void device*` is not legal MSL).
    auto globalsType = builder.createStructType();
    builder.addNameHintDecoration(globalsType, UnownedStringSlice("slang_RTGlobals"));
    addField(globalsType, "reserved", uintType);

    // `slang_RTSbt` (section 4.4): the software shader binding table — region
    // base indices into the visible function table plus the per-instance
    // contribution (DXR's InstanceContributionToHitGroupIndex), which Metal's
    // instance descriptor cannot carry for the visible table. The
    // `callableBase` and `hitStride` fields are part of the ABI but have no
    // compiler-side reader until callable shaders (phase P2) land.
    auto sbtType = builder.createStructType();
    builder.addNameHintDecoration(sbtType, UnownedStringSlice("slang_RTSbt"));
    context.missBaseKey = addField(sbtType, "missBase", uintType);
    context.hitBaseKey = addField(sbtType, "hitBase", uintType);
    addField(sbtType, "callableBase", uintType);
    addField(sbtType, "hitStride", uintType);
    context.instanceSbtOffsetsKey =
        addField(sbtType, "instanceSbtOffsets", builder.getPtrType(uintType, AddressSpace::Global));

    context.ctxPtrType = builder.getPtrType(ctxType, AddressSpace::ThreadLocal);
    context.globalsPtrType = builder.getPtrType(globalsType, AddressSpace::Global);
    context.sbtPtrType = builder.getPtrType(sbtType, AddressSpace::Uniform);

    IRType* handlerParamTypes[] = {context.ctxPtrType, context.globalsPtrType};
    context.handlerFuncType = builder.getFuncType(2, handlerParamTypes, builder.getVoidType());

    IRInst* tableOperand = context.handlerFuncType;
    context.visibleTableType =
        (IRType*)builder.getType(kIROp_MetalVisibleFunctionTableType, 1, &tableOperand);
}

/// Return the Metal attribute string `buffer(<index>)`.
static String getMetalBufferAttr(int index)
{
    StringBuilder sb;
    sb << "buffer(" << index << ")";
    return sb.produceString();
}

/// Create an entry-point parameter, appended after any existing parameters of
/// `func`, carrying a name hint and a Metal attribute (`[[buffer(n)]]`,
/// `[[thread_position_in_grid]]`, ...) via a target-system-value decoration.
static IRParam* addSystemParam(
    IRBuilder& builder,
    IRFunc* func,
    IRType* type,
    const char* name,
    const String& metalAttribute)
{
    auto firstBlock = func->getFirstBlock();
    auto param = builder.createParam(type);
    if (auto firstOrdinary = firstBlock->getFirstOrdinaryInst())
        param->insertBefore(firstOrdinary);
    else
        param->insertAtEnd(firstBlock);
    builder.addNameHintDecoration(param, UnownedStringSlice(name));
    builder.addTargetSystemValueDecoration(param, metalAttribute.getUnownedSlice());
    return param;
}

/// Return the address of `ctx->payload` reinterpreted as a pointer to
/// `payloadType`, diagnosing a payload that does not fit the blob. This is
/// the pack/unpack bridge of the ABI: every trace site and every handler
/// reads and writes the payload through this typed view of the same bytes.
static IRInst* emitTypedPayloadBlobAddr(
    MetalRayTracingLegalizationContext& context,
    IRBuilder& builder,
    IRInst* ctxPtr,
    IRType* payloadType,
    IRInst* diagnosticInst)
{
    IRSizeAndAlignment sizeAndAlignment;
    if (SLANG_SUCCEEDED(getNaturalSizeAndAlignment(
            context.targetProgram->getTargetReq(),
            payloadType,
            &sizeAndAlignment)))
    {
        if (sizeAndAlignment.size > kMetalRTMaxPayloadSize &&
            context.diagnosedOversizePayloadTypes.add(payloadType))
        {
            context.sink->diagnose(Diagnostics::MetalRaytracingPayloadTooLarge{
                .payloadType = payloadType,
                .payloadSize = String(sizeAndAlignment.size),
                .maxSize = String(kMetalRTMaxPayloadSize),
                .location = getDiagnosticPos(diagnosticInst)});
        }
    }
    else
    {
        // A ray payload is plain shader-memory data by construction (it is
        // stored to and loaded from a byte blob on every target), so a type
        // with no natural layout should be impossible here.
        SLANG_ASSERT(!"Metal ray payload type has no natural layout");
    }
    auto blobAddr = builder.emitFieldAddress(ctxPtr, context.payloadKey);
    return builder.emitBitCast(
        builder.getPtrType(payloadType, AddressSpace::ThreadLocal),
        blobAddr);
}

/// Rewrite a ray-generation entry point into the compute kernel of the Metal
/// execution model: it keeps its user parameters (the acceleration structure
/// and other resources stay ordinary Metal bindings), gains the implicit
/// system parameters of section 4.4, and owns the thread-local
/// `slang_RTContext` whose launch fields are filled from the compute dispatch
/// coordinates.
static void processRayGenerationEntryPoint(
    MetalRayTracingLegalizationContext& context,
    IRFunc* func)
{
    ensureSharedTypes(context);

    IRBuilder builder(context.module);
    auto firstBlock = func->getFirstBlock();
    SLANG_ASSERT(firstBlock);

    auto uint3Type = builder.getVectorType(builder.getUIntType(), 3);

    auto handlerTable = addSystemParam(
        builder,
        func,
        context.visibleTableType,
        "slang_rtHandlers",
        getMetalBufferAttr(kMetalRTHandlersBufferIndex));
    auto sbtParam = addSystemParam(
        builder,
        func,
        context.sbtPtrType,
        "slang_rtSbt",
        getMetalBufferAttr(kMetalRTSbtBufferIndex));
    auto globalsParam = addSystemParam(
        builder,
        func,
        context.globalsPtrType,
        "slang_rtGlobals",
        getMetalBufferAttr(kMetalRTGlobalsBufferIndex));
    auto tidParam = addSystemParam(
        builder,
        func,
        uint3Type,
        "slang_rtLaunchIndex",
        String("thread_position_in_grid"));
    auto tdimParam =
        addSystemParam(builder, func, uint3Type, "slang_rtLaunchDim", String("threads_per_grid"));

    // The context lives in the kernel's thread address space and is threaded
    // by pointer through every handler call.
    builder.setInsertBefore(firstBlock->getFirstOrdinaryInst());
    auto ctxVar = builder.emitVar(context.ctxStructType);
    builder.addNameHintDecoration(ctxVar, UnownedStringSlice("slang_rtCtx"));

    builder.emitStore(builder.emitFieldAddress(ctxVar, context.launchIndexKey), tidParam);
    builder.emitStore(builder.emitFieldAddress(ctxVar, context.launchDimKey), tdimParam);

    fixUpFuncType(func);

    MetalRayTracingLegalizationContext::RTEntryInfo info;
    info.stage = Stage::RayGeneration;
    info.ctxPtr = ctxVar;
    info.globalsPtr = globalsParam;
    info.handlerTable = handlerTable;
    info.sbtPtr = sbtParam;
    context.rtEntries[func] = info;
}

/// Replace a handler's payload parameter (`inout` in the user's signature)
/// with a local variable that is unpacked from the context's payload blob.
/// Returns the typed view of the blob so the caller can pack the variable
/// back at every exit point.
static IRInst* bridgePayloadParamThroughContextBlob(
    MetalRayTracingLegalizationContext& context,
    IRBuilder& builder,
    IRInst* ctxParam,
    IRParam* param,
    IROutParamTypeBase* paramType,
    IRInst*& outPayloadVar)
{
    auto payloadType = (IRType*)paramType->getValueType();
    auto typedBlobAddr = emitTypedPayloadBlobAddr(context, builder, ctxParam, payloadType, param);
    auto payloadVar = builder.emitVar(payloadType);
    builder.addNameHintDecoration(payloadVar, UnownedStringSlice("slang_rtPayload"));
    builder.emitStore(payloadVar, builder.emitLoad(typedBlobAddr));
    param->replaceUsesWith(payloadVar);
    outPayloadVar = payloadVar;
    return typedBlobAddr;
}

/// Replace a closest-hit shader's intersection-attribute parameter with a
/// value rebuilt from the committed barycentrics that the trace lowering
/// stored in the context. The parameter arrives either by value or wrapped
/// in a borrow-in (pointer) parameter type, depending on how the front end
/// lowered the `in` parameter. Only the built-in single-`float2` attribute
/// shape exists for triangle geometry (procedural attributes arrive with
/// intersection-function support, phase P1); anything else is diagnosed.
/// Returns false if the parameter could not be replaced.
static bool bridgeTriangleAttributeParam(
    MetalRayTracingLegalizationContext& context,
    IRBuilder& builder,
    IRInst* ctxParam,
    IRParam* param)
{
    auto attrType = param->getDataType();
    bool isPointerParam = false;
    if (auto ptrLikeType = as<IRPtrTypeBase>(attrType))
    {
        attrType = (IRType*)ptrLikeType->getValueType();
        isPointerParam = true;
    }

    auto attrStructType = as<IRStructType>(attrType);
    IRStructField* baryField = nullptr;
    int fieldCount = 0;
    if (attrStructType)
    {
        for (auto field : attrStructType->getFields())
        {
            baryField = field;
            fieldCount++;
        }
    }
    auto isFloat2 = [&](IRType* type)
    {
        auto vectorType = as<IRVectorType>(type);
        if (!vectorType)
            return false;
        if (vectorType->getElementType()->getOp() != kIROp_FloatType)
            return false;
        auto countLit = as<IRIntLit>(vectorType->getElementCount());
        return countLit && countLit->getValue() == 2;
    };
    if (!attrStructType || fieldCount != 1 || !isFloat2(baryField->getFieldType()))
    {
        context.sink->diagnose(Diagnostics::MetalRaytracingAttributeTypeNotSupported{
            .attributeType = attrType,
            .location = getDiagnosticPos(param)});
        return false;
    }

    auto bary = builder.emitLoad(builder.emitFieldAddress(ctxParam, context.triBarycentricsKey));
    IRInst* attrArgs[] = {bary};
    auto attrValue = builder.emitMakeStruct(attrType, 1, attrArgs);
    if (isPointerParam)
    {
        // Uses expect an address: materialize the value in a local.
        auto attrVar = builder.emitVar(attrType);
        builder.emitStore(attrVar, attrValue);
        param->replaceUsesWith(attrVar);
    }
    else
    {
        param->replaceUsesWith(attrValue);
    }
    return true;
}

/// Rewrite a miss or closest-hit entry point to the uniform visible-function
/// signature `void(slang_RTContext thread*, slang_RTGlobals device*)`. The
/// user's payload parameter becomes a local variable that is unpacked from the
/// context's payload blob on entry and packed back before every return; a
/// closest-hit attribute parameter is reconstructed from the committed-hit
/// barycentrics that the trace lowering stored in the context.
static void processHandlerEntryPoint(
    MetalRayTracingLegalizationContext& context,
    IRFunc* func,
    Stage stage)
{
    ensureSharedTypes(context);

    IRBuilder builder(context.module);
    auto firstBlock = func->getFirstBlock();
    SLANG_ASSERT(firstBlock);

    // Collect the original (DXR-signature) parameters before adding ours.
    List<IRParam*> oldParams;
    for (auto param = firstBlock->getFirstParam(); param; param = param->getNextParam())
        oldParams.add(param);

    auto insertBeforeInst = firstBlock->getFirstOrdinaryInst();
    auto ctxParam = builder.createParam(context.ctxPtrType);
    auto globalsParam = builder.createParam(context.globalsPtrType);
    if (insertBeforeInst)
    {
        ctxParam->insertBefore(insertBeforeInst);
        globalsParam->insertBefore(insertBeforeInst);
    }
    else
    {
        ctxParam->insertAtEnd(firstBlock);
        globalsParam->insertAtEnd(firstBlock);
    }
    builder.addNameHintDecoration(ctxParam, UnownedStringSlice("slang_rtCtx"));
    builder.addNameHintDecoration(globalsParam, UnownedStringSlice("slang_rtGlobals"));

    builder.setInsertBefore(insertBeforeInst);

    IRInst* typedPayloadBlobAddr = nullptr;
    IRInst* payloadVar = nullptr;
    List<IRParam*> paramsToRemove;
    for (auto param : oldParams)
    {
        bool replaced = false;
        if (auto outParamType = as<IROutParamTypeBase>(param->getDataType()))
        {
            typedPayloadBlobAddr = bridgePayloadParamThroughContextBlob(
                context,
                builder,
                ctxParam,
                param,
                outParamType,
                payloadVar);
            replaced = true;
        }
        else
        {
            replaced = bridgeTriangleAttributeParam(context, builder, ctxParam, param);
        }
        if (replaced)
            paramsToRemove.add(param);
    }

    // Pack the payload back into the blob at every exit point, so the trace
    // site in the kernel observes the handler's writes.
    if (payloadVar)
    {
        for (auto block : func->getBlocks())
        {
            auto returnInst = as<IRReturn>(block->getTerminator());
            if (!returnInst)
                continue;
            builder.setInsertBefore(returnInst);
            builder.emitStore(typedPayloadBlobAddr, builder.emitLoad(payloadVar));
        }
    }

    for (auto param : paramsToRemove)
        param->removeAndDeallocate();

    fixUpFuncType(func);

    MetalRayTracingLegalizationContext::RTEntryInfo info;
    info.stage = stage;
    info.ctxPtr = ctxParam;
    info.globalsPtr = globalsParam;
    context.rtEntries[func] = info;
}

/// Diagnose references to globally bound shader parameters or mutable global
/// variables from a miss/closest-hit handler, including references made by
/// helper functions the handler calls. Visible functions cannot see the
/// kernel's resource bindings, and Metal has no module-scope mutable state;
/// until the `slang_RTGlobals` hoisting of section 4.3 is implemented, such
/// references cannot be honored and must be rejected loudly rather than
/// emitted as silently broken MSL. The check walks the direct-call graph
/// because after `inlineMetalRTOpsIntoEntryPoints` only helpers *without*
/// ray-tracing ops remain outlined, and those can still touch globals
/// (`introduceExplicitGlobalContext` would later thread a kernel-context
/// parameter through them into the handler, corrupting the fixed
/// visible-function ABI — that pass carries a backstop diagnostic for the
/// same condition).
static void checkForGlobalStateInHandler(
    MetalRayTracingLegalizationContext& context,
    IRFunc* entryPointFunc)
{
    List<IRFunc*> workList;
    HashSet<IRFunc*> reachableFuncs;
    workList.add(entryPointFunc);
    reachableFuncs.add(entryPointFunc);

    HashSet<IRInst*> diagnosed;
    for (Index i = 0; i < workList.getCount(); i++)
    {
        auto func = workList[i];
        for (auto block : func->getBlocks())
        {
            for (auto inst : block->getChildren())
            {
                if (auto call = as<IRCall>(inst))
                {
                    if (auto callee = as<IRFunc>(call->getCallee()))
                    {
                        if (reachableFuncs.add(callee))
                            workList.add(callee);
                    }
                }
                for (UInt j = 0; j < inst->getOperandCount(); j++)
                {
                    auto operand = inst->getOperand(j);
                    bool isGlobalState = as<IRGlobalParam>(operand) || as<IRGlobalVar>(operand);
                    if (isGlobalState && diagnosed.add(operand))
                    {
                        context.sink->diagnose(Diagnostics::MetalRaytracingGlobalParamInHandler{
                            .paramName = operand,
                            .entryPointName = entryPointFunc,
                            .location = getDiagnosticPos(inst)});
                    }
                }
            }
        }
    }
}

/// Diagnose user resources whose Metal buffer binding collides with the
/// fixed indices reserved for the ray-generation kernel's system parameters
/// (`buffer(28)`..`buffer(30)`). User bindings are laid out from index 0
/// upward, so a collision requires a very large binding range or an explicit
/// user-chosen register — either way it must fail loudly, because the kernel
/// would otherwise bind two resources to one slot.
static void checkForReservedBufferIndexCollisions(MetalRayTracingLegalizationContext& context)
{
    for (auto inst : context.module->getGlobalInsts())
    {
        auto globalParam = as<IRGlobalParam>(inst);
        if (!globalParam)
            continue;
        auto layoutDecor = globalParam->findDecoration<IRLayoutDecoration>();
        if (!layoutDecor)
            continue;
        auto varLayout = as<IRVarLayout>(layoutDecor->getLayout());
        if (!varLayout)
            continue;
        auto offsetAttr = varLayout->findOffsetAttr(LayoutResourceKind::MetalBuffer);
        if (!offsetAttr)
            continue;
        auto offset = offsetAttr->getOffset();
        if (offset >= kMetalRTGlobalsBufferIndex)
        {
            context.sink->diagnose(Diagnostics::MetalRaytracingBindingCollision{
                .paramName = globalParam,
                .bufferIndex = String(offset),
                .location = getDiagnosticPos(globalParam)});
        }
    }
}

/// Store `value` into the context field named by `key`.
static void storeContextField(IRBuilder& builder, IRInst* ctxPtr, IRStructKey* key, IRInst* value)
{
    builder.emitStore(builder.emitFieldAddress(ctxPtr, key), value);
}

/// Store the state of the ray being traced into the context, common to the
/// hit and miss dispatch paths of one `TraceRay` expansion.
static void storeTraceRayStateIntoContext(
    MetalRayTracingLegalizationContext& context,
    IRBuilder& builder,
    IRInst* ctxPtr,
    IRInst* origin,
    IRInst* tMin,
    IRInst* direction,
    IRInst* tMax,
    IRInst* rayFlags)
{
    storeContextField(builder, ctxPtr, context.originKey, origin);
    storeContextField(builder, ctxPtr, context.tMinKey, tMin);
    storeContextField(builder, ctxPtr, context.directionKey, direction);
    storeContextField(builder, ctxPtr, context.tMaxKey, tMax);
    storeContextField(builder, ctxPtr, context.rayFlagsKey, rayFlags);
}

/// Pack the caller's payload into the context blob, call the handler record
/// selected by `recordIndex` through the visible function table, and unpack
/// the blob back into the caller's payload variable — the shared tail of the
/// hit and miss dispatch paths.
static void emitHandlerDispatch(
    MetalRayTracingLegalizationContext& context,
    MetalRayTracingLegalizationContext::RTEntryInfo& entryInfo,
    IRBuilder& builder,
    IRInst* recordIndex,
    IRInst* payloadPtr,
    IRInst* diagnosticInst)
{
    auto payloadType = (IRType*)as<IRPtrTypeBase>(payloadPtr->getDataType())->getValueType();
    auto typedBlobAddr =
        emitTypedPayloadBlobAddr(context, builder, entryInfo.ctxPtr, payloadType, diagnosticInst);
    builder.emitStore(typedBlobAddr, builder.emitLoad(payloadPtr));

    IRInst* callArgs[] =
        {entryInfo.handlerTable, recordIndex, entryInfo.ctxPtr, entryInfo.globalsPtr};
    builder.emitIntrinsicInst(builder.getVoidType(), kIROp_MetalRTHandlerCall, 4, callArgs);

    builder.emitStore(payloadPtr, builder.emitLoad(typedBlobAddr));
}

/// Lower a `metalRTCallHitHandler` op inside a ray-generation kernel: store
/// the ray and committed-hit state into the context, then dispatch the
/// hit-group record computed by the DXR indexing formula
///   hitBase + RayContributionToHitGroupIndex
///           + MultiplierForGeometryContributionToHitGroupIndex * geometryIndex
///           + instanceSbtOffsets[instanceIndex]
/// with the per-instance contribution supplied through the SBT buffer,
/// because Metal's instance descriptor offset only applies to intersection
/// function tables, not visible function tables.
static void lowerCallHitHandlerOp(
    MetalRayTracingLegalizationContext& context,
    MetalRayTracingLegalizationContext::RTEntryInfo& entryInfo,
    IRMetalRTCallHitHandler* inst)
{
    IRBuilder builder(context.module);
    builder.setInsertBefore(inst);

    auto uintType = builder.getUIntType();
    auto ctxPtr = entryInfo.ctxPtr;

    storeTraceRayStateIntoContext(
        context,
        builder,
        ctxPtr,
        inst->getOrigin(),
        inst->getTMin(),
        inst->getDirection(),
        inst->getTMax(),
        inst->getRayFlags());
    storeContextField(builder, ctxPtr, context.hitTKey, inst->getHitT());
    storeContextField(builder, ctxPtr, context.instanceIndexKey, inst->getInstanceIndex());
    storeContextField(builder, ctxPtr, context.instanceIDKey, inst->getInstanceID());
    storeContextField(builder, ctxPtr, context.geometryIndexKey, inst->getGeometryIndex());
    storeContextField(builder, ctxPtr, context.primitiveIndexKey, inst->getPrimitiveIndex());
    storeContextField(builder, ctxPtr, context.triBarycentricsKey, inst->getBarycentrics());

    auto frontKind = builder.getIntValue(uintType, kHitKindTriangleFrontFace);
    auto backKind = builder.getIntValue(uintType, kHitKindTriangleBackFace);
    IRInst* selectArgs[] = {inst->getIsFrontFace(), frontKind, backKind};
    auto hitKind = builder.emitIntrinsicInst(uintType, kIROp_Select, 3, selectArgs);
    storeContextField(builder, ctxPtr, context.hitKindKey, hitKind);

    auto hitBase = builder.emitLoad(builder.emitFieldAddress(entryInfo.sbtPtr, context.hitBaseKey));
    auto offsetsPtr =
        builder.emitLoad(builder.emitFieldAddress(entryInfo.sbtPtr, context.instanceSbtOffsetsKey));
    auto instanceContribution =
        builder.emitLoad(builder.emitGetOffsetPtr(offsetsPtr, inst->getInstanceIndex()));

    auto recordIndex =
        builder.emitAdd(uintType, hitBase, inst->getRayContributionToHitGroupIndex());
    recordIndex = builder.emitAdd(
        uintType,
        recordIndex,
        builder.emitMul(
            uintType,
            inst->getMultiplierForGeometryContribution(),
            inst->getGeometryIndex()));
    recordIndex = builder.emitAdd(uintType, recordIndex, instanceContribution);

    emitHandlerDispatch(context, entryInfo, builder, recordIndex, inst->getPayloadPtr(), inst);
    inst->removeAndDeallocate();
}

/// Lower a `metalRTCallMissHandler` op inside a ray-generation kernel: store
/// the ray state into the context and dispatch the miss record
/// `missBase + missShaderIndex`.
static void lowerCallMissHandlerOp(
    MetalRayTracingLegalizationContext& context,
    MetalRayTracingLegalizationContext::RTEntryInfo& entryInfo,
    IRMetalRTCallMissHandler* inst)
{
    IRBuilder builder(context.module);
    builder.setInsertBefore(inst);

    auto uintType = builder.getUIntType();
    auto ctxPtr = entryInfo.ctxPtr;

    storeTraceRayStateIntoContext(
        context,
        builder,
        ctxPtr,
        inst->getOrigin(),
        inst->getTMin(),
        inst->getDirection(),
        inst->getTMax(),
        inst->getRayFlags());
    // In a miss shader DXR defines RayTCurrent() as the ray's TMax.
    storeContextField(builder, ctxPtr, context.hitTKey, inst->getTMax());

    auto missBase =
        builder.emitLoad(builder.emitFieldAddress(entryInfo.sbtPtr, context.missBaseKey));
    auto recordIndex = builder.emitAdd(uintType, missBase, inst->getMissShaderIndex());

    emitHandlerDispatch(context, entryInfo, builder, recordIndex, inst->getPayloadPtr(), inst);
    inst->removeAndDeallocate();
}

/// Lower every transient `metalRT*` op inside a rewritten ray-tracing entry
/// point: system-value readers become loads of context fields, and the
/// call-handler ops become the section 5 trace-dispatch sequence.
static void lowerMetalRTOpsInFunc(
    MetalRayTracingLegalizationContext& context,
    IRFunc* func,
    MetalRayTracingLegalizationContext::RTEntryInfo& entryInfo)
{
    List<IRInst*> workList;
    for (auto block : func->getBlocks())
        for (auto inst : block->getChildren())
            if (isMetalRTOp(inst))
                workList.add(inst);

    for (auto inst : workList)
    {
        switch (inst->getOp())
        {
        case kIROp_MetalRTCallMissHandler:
        case kIROp_MetalRTCallHitHandler:
            {
                // Tracing needs the table/SBT system parameters, which only
                // the ray-generation kernel receives; a trace reaching any
                // other stage (e.g. recursion from a closest-hit shader) is
                // not supported yet (phase P3 of the design doc).
                if (entryInfo.stage != Stage::RayGeneration)
                {
                    context.sink->diagnose(Diagnostics::MetalRaytracingTraceOutsideRaygen{
                        .location = getDiagnosticPos(inst)});
                    inst->removeAndDeallocate();
                }
                else if (auto hitCall = as<IRMetalRTCallHitHandler>(inst))
                {
                    lowerCallHitHandlerOp(context, entryInfo, hitCall);
                }
                else
                {
                    lowerCallMissHandlerOp(
                        context,
                        entryInfo,
                        cast<IRMetalRTCallMissHandler>(inst));
                }
            }
            break;
        default:
            {
                auto key = getContextKeyForReaderOp(context, inst->getOp());
                SLANG_ASSERT(key);
                IRBuilder builder(context.module);
                builder.setInsertBefore(inst);
                auto value = builder.emitLoad(builder.emitFieldAddress(entryInfo.ctxPtr, key));
                inst->replaceUsesWith(value);
                inst->removeAndDeallocate();
            }
            break;
        }
    }
}

/// Diagnose any `metalRT*` op that survived: after inlining, every such op
/// must sit inside a supported, rewritten ray-tracing entry point. Ops can
/// legitimately remain only in unreferenced functions (dead code that later
/// passes remove), which are skipped. One diagnostic per module suffices —
/// this is a should-never-happen guard, not a user-facing report of every
/// offending site.
static void diagnoseStrayMetalRTOps(MetalRayTracingLegalizationContext& context)
{
    for (auto inst : context.module->getGlobalInsts())
    {
        auto func = as<IRFunc>(inst);
        if (!func)
            continue;
        if (context.rtEntries.containsKey(func))
            continue;
        bool isLive = func->findDecoration<IREntryPointDecoration>() ||
                      func->findDecoration<IRKeepAliveDecoration>() ||
                      func->findDecoration<IRPublicDecoration>() || func->hasUses();
        if (!isLive)
            continue;
        for (auto block : func->getBlocks())
        {
            for (auto opInst : block->getChildren())
            {
                if (isMetalRTOp(opInst))
                {
                    context.sink->diagnose(Diagnostics::MetalRaytracingIntrinsicOutsideEntryPoint{
                        .location = getDiagnosticPos(opInst)});
                    return;
                }
            }
        }
    }
}

void legalizeMetalRayTracing(
    IRModule* module,
    TargetProgram* targetProgram,
    DiagnosticSink* sink,
    List<EntryPointInfo>& entryPoints)
{
    MetalRayTracingLegalizationContext context;
    context.module = module;
    context.targetProgram = targetProgram;
    context.sink = sink;

    // Partition the ray-tracing entry points by stage, diagnosing the stages
    // that later phases of docs/design/metal-raytracing.md will add.
    List<EntryPointInfo> rayGenEntryPoints;
    List<EntryPointInfo> handlerEntryPoints;
    bool anyRTEntryPoints = false;
    for (auto& entryPoint : entryPoints)
    {
        auto stage = entryPoint.entryPointDecor->getProfile().getStage();
        switch (stage)
        {
        case Stage::RayGeneration:
            anyRTEntryPoints = true;
            rayGenEntryPoints.add(entryPoint);
            break;
        case Stage::Miss:
        case Stage::ClosestHit:
            anyRTEntryPoints = true;
            handlerEntryPoints.add(entryPoint);
            break;
        case Stage::AnyHit:
        case Stage::Intersection:
        case Stage::Callable:
            anyRTEntryPoints = true;
            sink->diagnose(Diagnostics::MetalRaytracingStageNotSupported{
                .stageName = String(getStageName(stage)),
                .location = getDiagnosticPos(entryPoint.entryPointFunc)});
            break;
        default:
            break;
        }
    }
    if (!anyRTEntryPoints)
        return;

    // The raygen kernel's system parameters live at fixed buffer indices;
    // user bindings must stay below them.
    if (rayGenEntryPoints.getCount() != 0)
        checkForReservedBufferIndexCollisions(context);

    inlineMetalRTOpsIntoEntryPoints(module);

    for (auto& entryPoint : rayGenEntryPoints)
        processRayGenerationEntryPoint(context, entryPoint.entryPointFunc);
    for (auto& entryPoint : handlerEntryPoints)
    {
        processHandlerEntryPoint(
            context,
            entryPoint.entryPointFunc,
            entryPoint.entryPointDecor->getProfile().getStage());
        checkForGlobalStateInHandler(context, entryPoint.entryPointFunc);
    }

    for (auto& [func, info] : context.rtEntries)
        lowerMetalRTOpsInFunc(context, func, info);

    diagnoseStrayMetalRTOps(context);
}

} // namespace Slang
