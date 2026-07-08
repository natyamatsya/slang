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
// semantics). The sizes match DXR driver conventions; exceeding either blob
// is a compile-time error.
static const IRIntegerValue kMetalRTMaxPayloadSize = 64;
static const IRIntegerValue kMetalRTMaxAttributeSize = 32;

// Fixed Metal buffer binding indices for the implicit system parameters that
// a ray-generation kernel receives. These sit at the top of Metal's [[buffer]]
// index range (max 30) so they cannot collide with normally laid-out user
// parameters, which are assigned from 0 upward. The intersection function
// table is only bound when the program links anyhit/intersection stages. The
// runtime contract is documented in docs/design/metal-raytracing.md section
// 4.4.
static const int kMetalRTIsectBufferIndex = 27;
static const int kMetalRTGlobalsBufferIndex = 28;
static const int kMetalRTSbtBufferIndex = 29;
static const int kMetalRTHandlersBufferIndex = 30;

// DXR hit-kind values for fixed-function triangle hits.
static const int kHitKindTriangleFrontFace = 254;
static const int kHitKindTriangleBackFace = 255;

// The committed-intersection status codes returned by the `_slang_rtTrace`
// prelude helper (mirroring `metal::raytracing::intersection_type`).
static const int kMetalRTCommittedNone = 0;

// The DXR system-value reader ops and the `slang_RTContext` field each one
// loads. This table is the single source of truth relating the two: it
// expands into both the op classification (`isMetalRTOp`) and the lowering
// (`getContextKeyForReaderOp`). Inside anyhit and intersection functions,
// where Metal supplies the candidate state as tagged parameters, a per-entry
// override table takes precedence over the context field.
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

    // Set when the module contains anyhit or intersection entry points: the
    // ray-generation kernel then receives the `slang_rtIsect` intersection
    // function table and threads it into traversal.
    bool hasIntersectionStages = false;

    // The shared ABI types, created on first use by `ensureSharedTypes`.
    IRStructType* ctxStructType = nullptr;
    IRStructType* globalsStructType = nullptr;
    IRType* ctxPtrType = nullptr;        // slang_RTContext thread*
    IRType* ctxRayDataPtrType = nullptr; // slang_RTContext ray_data* ([[payload]])
    IRType* globalsPtrType = nullptr;
    IRType* sbtPtrType = nullptr;
    IRType* visibleTableType = nullptr;
    IRType* isectTableType = nullptr;

    // The return type of a Metal bounding-box intersection function:
    // { bool accept [[accept_intersection]]; float distance [[distance]]; }.
    IRStructType* bboxResultType = nullptr;

    // Field keys of `slang_RTContext` that the lowering reads or writes.
    // (The struct has more fields than this; keys are only kept for the
    // fields this pass touches.)
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
    IRStructKey* attributesKey = nullptr;
    IRStructKey* payloadKey = nullptr;
    IRStructKey* launchIndexKey = nullptr;
    IRStructKey* launchDimKey = nullptr;

    // Field keys of `slang_RTSbt` that the dispatch lowerings read.
    IRStructKey* missBaseKey = nullptr;
    IRStructKey* hitBaseKey = nullptr;
    IRStructKey* callableBaseKey = nullptr;
    IRStructKey* instanceSbtOffsetsKey = nullptr;

    // Field keys of `slang_RTGlobals`: the fixed dispatch-state header,
    // plus one field per hoisted user global (keyed by the global param).
    IRStructKey* globalsHandlersKey = nullptr;
    IRStructKey* globalsIsectKey = nullptr;
    IRStructKey* globalsSbtKey = nullptr;
    Dictionary<IRInst*, IRStructKey*> hoistedGlobalFields;

    // Per-entry-point access paths to the ABI values: the context pointer is
    // a local variable in a ray-generation kernel, the first parameter of a
    // miss/closest-hit/callable handler, and the `[[payload]]` parameter of
    // an anyhit/intersection function. The table/SBT handles come from the
    // kernel's system parameters or, in a handler, from the globals
    // argument buffer; they are null in anyhit/intersection functions,
    // which cannot dispatch.
    struct RTEntryInfo
    {
        Stage stage = Stage::Unknown;
        IRInst* ctxPtr = nullptr;
        IRInst* globalsPtr = nullptr;
        IRInst* handlerTable = nullptr;
        IRInst* sbtPtr = nullptr;
        IRInst* isectTable = nullptr;

        // In anyhit/intersection functions, DXR system values come from
        // Metal's tagged parameters (`[[distance]]`, `[[primitive_id]]`,
        // ...) rather than context fields; this maps a reader opcode to the
        // value that replaces it.
        Dictionary<int, IRInst*> readerOverrides;

        // The `[[min_distance]]` / `[[max_distance]]` parameters of an
        // intersection function, needed by the ReportHit range guard.
        IRInst* isectMinT = nullptr;
        IRInst* isectMaxT = nullptr;
    };
    Dictionary<IRFunc*, RTEntryInfo> rtEntries;

    // Types already reported as exceeding their blob, so one oversized type
    // diagnoses once even though it is bridged at several points.
    HashSet<IRType*> diagnosedOversizePayloadTypes;
    HashSet<IRType*> diagnosedOversizeAttributeTypes;
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
    case kIROp_MetalRTTraceRay:
    case kIROp_MetalRTCallShader:
    case kIROp_MetalRTReportHit:
    case kIROp_MetalRTIgnoreHit:
    case kIROp_MetalRTAcceptHitAndEndSearch:
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

/// Does any instruction of `func` reference a globally bound shader
/// parameter?
static bool funcReferencesGlobalParam(IRFunc* func)
{
    for (auto block : func->getBlocks())
        for (auto inst : block->getChildren())
            for (UInt i = 0; i < inst->getOperandCount(); i++)
                if (as<IRGlobalParam>(inst->getOperand(i)))
                    return true;
    return false;
}

/// Inline every call to a non-entry-point function that (transitively)
/// contains a Metal ray-tracing op or references a global shader parameter,
/// so that both end up physically inside their entry points: ray-tracing ops
/// need the context variable and system parameters in scope, and global
/// references inside handler-stage entries must be rewritten to
/// `slang_RTGlobals` fields (which only the entry can address). `TraceRay`
/// itself is `[ForceInline]`, so its body is already inlined into the user's
/// calling function; this handles user helper functions.
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
            if (!funcContainsMetalRTOp(func) && !funcReferencesGlobalParam(func))
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
    auto boolType = builder.getBoolType();
    auto float2Type = builder.getVectorType(floatType, 2);
    auto float3Type = builder.getVectorType(floatType, 3);
    auto uint3Type = builder.getVectorType(uintType, 3);
    auto byteType = builder.getBasicType(BaseType::UInt8);

    // The ABI struct and field names are pinned exactly (no uniqueness
    // suffixes) for two reasons: the `_slang_rtTrace` prelude template
    // accesses context fields by name, and separately compiled stage
    // libraries must emit byte-identical struct declarations so the kernel
    // and its visible/intersection functions agree on the layout.
    auto pinName = [&](IRInst* inst, const char* name)
    {
        builder.addNameHintDecoration(inst, UnownedStringSlice(name));
        builder.addExternCppDecoration(inst, UnownedStringSlice(name));
    };
    auto addField = [&](IRStructType* structType, const char* name, IRType* fieldType)
    {
        auto key = builder.createStructKey();
        pinName(key, name);
        builder.createStructField(structType, key, fieldType);
        return key;
    };

    // `slang_RTContext` (docs/design/metal-raytracing.md section 4.1): the
    // single thread-local struct through which all cross-stage communication
    // happens. The field order is part of the ABI; the kernel and the
    // visible/intersection functions are all emitted from this one type.
    auto ctxType = builder.createStructType();
    pinName(ctxType, "slang_RTContext");
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
    context.attributesKey = addField(
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

    // `slang_RTSbt` (section 4.4): the software shader binding table — region
    // base indices into the visible function table plus the per-instance
    // contribution (DXR's InstanceContributionToHitGroupIndex), which Metal's
    // instance descriptor cannot carry for the visible table. The
    // `hitStride` field is part of the ABI but has no compiler-side reader
    // yet (the hit-group multiplier arrives as a TraceRay argument).
    auto sbtType = builder.createStructType();
    pinName(sbtType, "slang_RTSbt");
    context.missBaseKey = addField(sbtType, "missBase", uintType);
    context.hitBaseKey = addField(sbtType, "hitBase", uintType);
    context.callableBaseKey = addField(sbtType, "callableBase", uintType);
    addField(sbtType, "hitStride", uintType);
    context.instanceSbtOffsetsKey = addField(
        sbtType,
        "instanceSbtOffsets",
        builder.getPtrType(uintType, AddressSpace::Global));

    // The return type of a bounding-box intersection function; the field
    // attributes are Metal's fixed protocol for accepting a candidate hit.
    auto bboxResultType = builder.createStructType();
    pinName(bboxResultType, "slang_RTBBoxResult");
    context.bboxResultType = bboxResultType;
    {
        auto acceptKey = addField(bboxResultType, "accept", boolType);
        builder.addTargetSystemValueDecoration(
            acceptKey,
            UnownedStringSlice("accept_intersection"));
        auto distanceKey = addField(bboxResultType, "distance", floatType);
        builder.addTargetSystemValueDecoration(distanceKey, UnownedStringSlice("distance"));
    }

    // `slang_RTGlobals` (section 4.3): the argument buffer that carries
    // everything a handler-stage function needs beyond the context — the
    // dispatch state (visible function table, intersection function table,
    // and software SBT) used when a handler itself traces or calls, followed
    // by the user resources that handler stages reference (hoisted by
    // `hoistHandlerGlobals`, in global declaration order). The struct is
    // created as an empty shell first: the visible-function-table field's
    // signature mentions `slang_RTGlobals*` itself, so the pointer and table
    // types must exist before the fields can be added.
    auto globalsType = builder.createStructType();
    pinName(globalsType, "slang_RTGlobals");
    context.globalsStructType = globalsType;

    context.ctxPtrType = builder.getPtrType(ctxType, AddressSpace::ThreadLocal);
    context.ctxRayDataPtrType = builder.getPtrType(ctxType, AddressSpace::MetalRayData);
    context.globalsPtrType = builder.getPtrType(globalsType, AddressSpace::Global);
    context.sbtPtrType = builder.getPtrType(sbtType, AddressSpace::Uniform);

    context.visibleTableType = (IRType*)builder.getType(kIROp_MetalVisibleFunctionTableType);
    context.isectTableType = (IRType*)builder.getType(kIROp_MetalIntersectionFunctionTableType);

    context.globalsHandlersKey = addField(globalsType, "handlers", context.visibleTableType);
    context.globalsIsectKey = addField(globalsType, "isect", context.isectTableType);
    context.globalsSbtKey = addField(globalsType, "sbt", sbtType);
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
/// `[[distance]]`, `[[payload]]`, ...) via a target-system-value decoration.
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

/// Collect the parameters of `func`'s entry block into a list, so they can
/// be rewritten while new parameters are being added.
static List<IRParam*> collectEntryPointParams(IRFunc* func)
{
    List<IRParam*> params;
    for (auto param = func->getFirstBlock()->getFirstParam(); param;
         param = param->getNextParam())
        params.add(param);
    return params;
}

/// Return the address of `ctx->payload` reinterpreted as a pointer to
/// `payloadType` in `addressSpace`, diagnosing a payload that does not fit
/// the blob. This is the pack/unpack bridge of the ABI: every trace site and
/// every handler reads and writes the payload through this typed view of the
/// same bytes.
static IRInst* emitTypedPayloadBlobAddr(
    MetalRayTracingLegalizationContext& context,
    IRBuilder& builder,
    IRInst* ctxPtr,
    IRType* payloadType,
    AddressSpace addressSpace,
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
    return builder.emitBitCast(builder.getPtrType(payloadType, addressSpace), blobAddr);
}

/// Return the address of `ctx->attributes` reinterpreted as a pointer to
/// `attrType` in `addressSpace`, diagnosing an attribute type that does not
/// fit the blob. Intersection functions write the blob through this view on
/// `ReportHit`, and closest-hit shaders with a non-triangle attribute type
/// read it back.
static IRInst* emitTypedAttributesBlobAddr(
    MetalRayTracingLegalizationContext& context,
    IRBuilder& builder,
    IRInst* ctxPtr,
    IRType* attrType,
    AddressSpace addressSpace,
    IRInst* diagnosticInst)
{
    IRSizeAndAlignment sizeAndAlignment;
    if (SLANG_SUCCEEDED(getNaturalSizeAndAlignment(
            context.targetProgram->getTargetReq(),
            attrType,
            &sizeAndAlignment)))
    {
        if (sizeAndAlignment.size > kMetalRTMaxAttributeSize &&
            context.diagnosedOversizeAttributeTypes.add(attrType))
        {
            context.sink->diagnose(Diagnostics::MetalRaytracingAttributesTooLarge{
                .attributeType = attrType,
                .attributeSize = String(sizeAndAlignment.size),
                .maxSize = String(kMetalRTMaxAttributeSize),
                .location = getDiagnosticPos(diagnosticInst)});
        }
    }
    else
    {
        SLANG_ASSERT(!"Metal intersection attribute type has no natural layout");
    }
    auto blobAddr = builder.emitFieldAddress(ctxPtr, context.attributesKey);
    return builder.emitBitCast(builder.getPtrType(attrType, addressSpace), blobAddr);
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
    IRParam* isectTableParam = nullptr;
    if (context.hasIntersectionStages)
    {
        isectTableParam = addSystemParam(
            builder,
            func,
            context.isectTableType,
            "slang_rtIsect",
            getMetalBufferAttr(kMetalRTIsectBufferIndex));
    }
    auto tidParam = addSystemParam(
        builder,
        func,
        uint3Type,
        "slang_rtLaunchIndex",
        String("thread_position_in_grid"));
    auto tdimParam = addSystemParam(
        builder,
        func,
        uint3Type,
        "slang_rtLaunchDim",
        String("threads_per_grid"));

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
    info.isectTable = isectTableParam;
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
    auto typedBlobAddr = emitTypedPayloadBlobAddr(
        context,
        builder,
        ctxParam,
        payloadType,
        AddressSpace::ThreadLocal,
        param);
    auto payloadVar = builder.emitVar(payloadType);
    builder.addNameHintDecoration(payloadVar, UnownedStringSlice("slang_rtPayload"));
    builder.emitStore(payloadVar, builder.emitLoad(typedBlobAddr));
    param->replaceUsesWith(payloadVar);
    outPayloadVar = payloadVar;
    return typedBlobAddr;
}

/// Is `type` the built-in triangle intersection attribute shape: a struct
/// holding a single `float2` (the barycentrics)?
static bool isTriangleAttributeStruct(IRType* type)
{
    auto structType = as<IRStructType>(type);
    if (!structType)
        return false;
    IRStructField* onlyField = nullptr;
    int fieldCount = 0;
    for (auto field : structType->getFields())
    {
        onlyField = field;
        fieldCount++;
    }
    if (fieldCount != 1)
        return false;
    auto vectorType = as<IRVectorType>(onlyField->getFieldType());
    if (!vectorType)
        return false;
    if (vectorType->getElementType()->getOp() != kIROp_FloatType)
        return false;
    auto countLit = as<IRIntLit>(vectorType->getElementCount());
    return countLit && countLit->getValue() == 2;
}

/// Replace an intersection-attribute parameter with `attrValue` (handling
/// the borrow-in pointer wrapping the front end may have applied to the `in`
/// parameter).
static void replaceAttributeParam(IRBuilder& builder, IRParam* param, IRInst* attrValue)
{
    if (as<IRPtrTypeBase>(param->getDataType()))
    {
        // Uses expect an address: materialize the value in a local.
        auto attrVar = builder.emitVar(attrValue->getDataType());
        builder.emitStore(attrVar, attrValue);
        param->replaceUsesWith(attrVar);
    }
    else
    {
        param->replaceUsesWith(attrValue);
    }
}

/// Create one `slang_RTGlobals` field per globally bound shader parameter
/// that a miss/closest-hit/callable entry point references, in global
/// declaration order. The order is part of the argument-buffer ABI: the
/// runtime encodes the hoisted user resources after the fixed dispatch-state
/// header (handlers, isect, sbt) in exactly this order. The kernel keeps its
/// ordinary bindings for the same resources; the application binds each
/// hoisted resource in both places.
static void hoistHandlerGlobals(
    MetalRayTracingLegalizationContext& context,
    List<EntryPointInfo>& handlerEntryPoints)
{
    HashSet<IRInst*> usedGlobals;
    for (auto& entryPoint : handlerEntryPoints)
    {
        for (auto block : entryPoint.entryPointFunc->getBlocks())
            for (auto inst : block->getChildren())
                for (UInt i = 0; i < inst->getOperandCount(); i++)
                    if (auto globalParam = as<IRGlobalParam>(inst->getOperand(i)))
                        usedGlobals.add(globalParam);
    }
    if (usedGlobals.getCount() == 0)
        return;

    ensureSharedTypes(context);
    IRBuilder builder(context.module);
    for (auto inst : context.module->getGlobalInsts())
    {
        auto globalParam = as<IRGlobalParam>(inst);
        if (!globalParam || !usedGlobals.contains(globalParam))
            continue;
        auto key = builder.createStructKey();
        if (auto nameHint = globalParam->findDecoration<IRNameHintDecoration>())
        {
            // Pin the field name to the user's parameter name, so separately
            // compiled stage libraries agree on the emitted struct and the
            // runtime can identify the field.
            builder.addNameHintDecoration(key, nameHint->getName());
            builder.addExternCppDecoration(key, nameHint->getName());
        }
        builder.createStructField(context.globalsStructType, key, globalParam->getFullType());
        context.hoistedGlobalFields[globalParam] = key;
    }
}

/// Rewrite references to hoisted user globals inside `func` to loads from
/// the globals argument buffer (section 4.3): a visible function cannot see
/// the kernel's bindings. After the predicate-driven inlining, every such
/// reference sits directly in the entry's body.
static void rewriteHoistedGlobalUses(
    MetalRayTracingLegalizationContext& context,
    IRBuilder& builder,
    IRFunc* func,
    IRInst* globalsParam)
{
    for (auto block : func->getBlocks())
    {
        for (auto bodyInst : block->getModifiableChildren())
        {
            for (UInt i = 0; i < bodyInst->getOperandCount(); i++)
            {
                IRStructKey* fieldKey = nullptr;
                if (!context.hoistedGlobalFields.tryGetValue(bodyInst->getOperand(i), fieldKey))
                    continue;
                builder.setInsertBefore(bodyInst);
                auto value = builder.emitLoad(builder.emitFieldAddress(globalsParam, fieldKey));
                bodyInst->setOperand(i, value);
            }
        }
    }
}

/// Rewrite a miss or closest-hit entry point to the uniform visible-function
/// signature `void(slang_RTContext thread*, slang_RTGlobals device*)`. The
/// user's payload parameter becomes a local variable that is unpacked from the
/// context's payload blob on entry and packed back before every return. A
/// closest-hit attribute parameter is reconstructed from the committed-hit
/// state: the built-in triangle attribute shape reads the barycentrics field
/// the trace lowering stored, and any other attribute struct reads the
/// attribute blob that the intersection function's `ReportHit` filled.
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
    auto oldParams = collectEntryPointParams(func);

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

    MetalRayTracingLegalizationContext::RTEntryInfo info;
    info.stage = stage;
    info.ctxPtr = ctxParam;
    info.globalsPtr = globalsParam;

    // Snapshot the DXR system values at entry: a nested TraceRay/CallShader
    // in this handler reuses the context as its dispatch scratch space, so
    // later reads must not observe the nested state. Unused snapshots are
    // removed by DCE.
#define CASE(OP, KEY)                        \
    info.readerOverrides[kIROp_##OP] =       \
        builder.emitLoad(builder.emitFieldAddress(ctxParam, context.KEY));
    SLANG_METAL_RT_SYSTEM_VALUE_OPS(CASE)
#undef CASE

    // Dispatch state for nested TraceRay/CallShader, read from the globals
    // argument buffer (section 5.3); also removed by DCE when the handler
    // does not dispatch.
    info.handlerTable =
        builder.emitLoad(builder.emitFieldAddress(globalsParam, context.globalsHandlersKey));
    info.sbtPtr = builder.emitFieldAddress(globalsParam, context.globalsSbtKey);
    if (context.hasIntersectionStages)
    {
        info.isectTable =
            builder.emitLoad(builder.emitFieldAddress(globalsParam, context.globalsIsectKey));
    }

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
            // The remaining (read-only) parameter of a closest-hit shader is
            // the intersection attribute struct.
            auto attrType = param->getDataType();
            if (auto ptrLikeType = as<IRPtrTypeBase>(attrType))
                attrType = (IRType*)ptrLikeType->getValueType();

            if (isTriangleAttributeStruct(attrType))
            {
                auto bary = builder.emitLoad(
                    builder.emitFieldAddress(ctxParam, context.triBarycentricsKey));
                IRInst* attrArgs[] = {bary};
                auto attrValue = builder.emitMakeStruct(attrType, 1, attrArgs);
                replaceAttributeParam(builder, param, attrValue);
                replaced = true;
            }
            else if (as<IRStructType>(attrType))
            {
                // Procedural-geometry attributes: read back the blob that
                // the intersection function's ReportHit stored.
                auto typedAttrAddr = emitTypedAttributesBlobAddr(
                    context,
                    builder,
                    ctxParam,
                    attrType,
                    AddressSpace::ThreadLocal,
                    param);
                auto attrValue = builder.emitLoad(typedAttrAddr);
                replaceAttributeParam(builder, param, attrValue);
                replaced = true;
            }
            else
            {
                context.sink->diagnose(Diagnostics::MetalRaytracingAttributeTypeNotSupported{
                    .attributeType = attrType,
                    .location = getDiagnosticPos(param)});
            }
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

    rewriteHoistedGlobalUses(context, builder, func, globalsParam);

    fixUpFuncType(func);

    context.rtEntries[func] = info;
}

/// Rewrite every `return` of `func` (which returns `void` in the user's
/// signature) into `return <makeResult()>`.
template<typename F>
static void rewriteVoidReturns(IRFunc* func, IRBuilder& builder, const F& makeResult)
{
    List<IRReturn*> returns;
    for (auto block : func->getBlocks())
        if (auto returnInst = as<IRReturn>(block->getTerminator()))
            returns.add(returnInst);
    for (auto returnInst : returns)
    {
        builder.setInsertBefore(returnInst);
        builder.emitReturn(makeResult(builder));
        returnInst->removeAndDeallocate();
    }
}

/// Create the tagged parameters common to Metal triangle (anyhit) and
/// bounding-box (intersection) intersection functions — the world-space ray
/// and the candidate identification ids — and wire the reader overrides
/// that map the DXR system-value intrinsics to them. The stage-specific
/// parameters (`[[distance]]`/`[[barycentric_coord]]`/`[[front_facing]]`
/// for anyhit, `[[min_distance]]`/`[[max_distance]]` for intersection) and
/// the trailing `[[payload]]` context parameter remain at the call sites.
static void addIntersectionFunctionCommonParams(
    MetalRayTracingLegalizationContext& context,
    IRBuilder& builder,
    IRFunc* func,
    MetalRayTracingLegalizationContext::RTEntryInfo& info)
{
    SLANG_UNUSED(context);
    auto uintType = builder.getUIntType();
    auto float3Type = builder.getVectorType(builder.getBasicType(BaseType::Float), 3);

    // Note that Metal's `[[origin]]`/`[[direction]]` are the OBJECT-space
    // ray inside intersection functions; `WorldRayOrigin()`/`WorldRayDirection()`
    // therefore read the world-space context fields the trace dispatch
    // stored (no override), and these parameters are reserved for future
    // `ObjectRayOrigin()`/`ObjectRayDirection()` support.
    addSystemParam(builder, func, float3Type, "slang_rtObjOrigin", String("origin"));
    addSystemParam(builder, func, float3Type, "slang_rtObjDirection", String("direction"));

    const struct
    {
        IROp readerOp;
        IRType* type;
        const char* name;
        const char* metalAttribute;
    } commonParams[] = {
        {kIROp_MetalRTPrimitiveIndex, uintType, "slang_rtPrimitiveIndex", "primitive_id"},
        {kIROp_MetalRTInstanceIndex, uintType, "slang_rtInstanceIndex", "instance_id"},
        {kIROp_MetalRTInstanceID, uintType, "slang_rtInstanceID", "user_instance_id"},
        {kIROp_MetalRTGeometryIndex, uintType, "slang_rtGeometryIndex", "geometry_id"},
    };
    for (auto& spec : commonParams)
    {
        auto param =
            addSystemParam(builder, func, spec.type, spec.name, String(spec.metalAttribute));
        info.readerOverrides[spec.readerOp] = param;
    }
}

/// Create the trailing `[[payload]]` context parameter of an
/// anyhit/intersection function and record it as the entry's context
/// pointer.
static IRInst* addIntersectionFunctionContextParam(
    MetalRayTracingLegalizationContext& context,
    IRBuilder& builder,
    IRFunc* func,
    MetalRayTracingLegalizationContext::RTEntryInfo& info)
{
    auto ctxParam = addSystemParam(
        builder,
        func,
        context.ctxRayDataPtrType,
        "slang_rtCtx",
        String("payload"));
    info.ctxPtr = ctxParam;
    return ctxParam;
}

/// Rewrite an anyhit entry point into a Metal triangle intersection
/// function: `bool` result (accept/reject of the builtin triangle
/// candidate), the candidate state as tagged parameters, and the context as
/// the `[[payload]]` reference. The payload is accessed directly through a
/// typed view of the context blob — no local copy — because DXR requires
/// payload writes from an anyhit shader to persist even when the hit is
/// ignored.
static void processAnyHitEntryPoint(MetalRayTracingLegalizationContext& context, IRFunc* func)
{
    ensureSharedTypes(context);

    IRBuilder builder(context.module);
    auto firstBlock = func->getFirstBlock();
    SLANG_ASSERT(firstBlock);

    auto oldParams = collectEntryPointParams(func);

    auto uintType = builder.getUIntType();
    auto floatType = builder.getBasicType(BaseType::Float);
    auto float2Type = builder.getVectorType(floatType, 2);

    MetalRayTracingLegalizationContext::RTEntryInfo info;
    info.stage = Stage::AnyHit;
    addIntersectionFunctionCommonParams(context, builder, func, info);
    auto distanceParam =
        addSystemParam(builder, func, floatType, "slang_rtCandidateT", String("distance"));
    auto baryParam =
        addSystemParam(builder, func, float2Type, "slang_rtBary", String("barycentric_coord"));
    auto frontFaceParam = addSystemParam(
        builder,
        func,
        builder.getBoolType(),
        "slang_rtFrontFace",
        String("front_facing"));
    auto ctxParam = addIntersectionFunctionContextParam(context, builder, func, info);
    info.readerOverrides[kIROp_MetalRTRayTCurrent] = distanceParam;

    builder.setInsertBefore(firstBlock->getFirstOrdinaryInst());

    // DXR hit kinds for the fixed-function triangle candidate.
    IRInst* hitKindArgs[] = {
        frontFaceParam,
        builder.getIntValue(uintType, kHitKindTriangleFrontFace),
        builder.getIntValue(uintType, kHitKindTriangleBackFace)};
    auto hitKind = builder.emitIntrinsicInst(uintType, kIROp_Select, 3, hitKindArgs);
    info.readerOverrides[kIROp_MetalRTHitKind] = hitKind;

    List<IRParam*> paramsToRemove;
    for (auto param : oldParams)
    {
        bool replaced = false;
        if (auto outParamType = as<IROutParamTypeBase>(param->getDataType()))
        {
            auto payloadType = (IRType*)outParamType->getValueType();
            auto typedBlobAddr = emitTypedPayloadBlobAddr(
                context,
                builder,
                ctxParam,
                payloadType,
                AddressSpace::MetalRayData,
                param);
            param->replaceUsesWith(typedBlobAddr);
            replaced = true;
        }
        else
        {
            // The anyhit attribute parameter: triangle candidates only carry
            // the built-in barycentrics.
            auto attrType = param->getDataType();
            if (auto ptrLikeType = as<IRPtrTypeBase>(attrType))
                attrType = (IRType*)ptrLikeType->getValueType();
            if (!isTriangleAttributeStruct(attrType))
            {
                context.sink->diagnose(Diagnostics::MetalRaytracingAttributeTypeNotSupported{
                    .attributeType = attrType,
                    .location = getDiagnosticPos(param)});
                continue;
            }
            IRInst* attrArgs[] = {baryParam};
            auto attrValue = builder.emitMakeStruct(attrType, 1, attrArgs);
            replaceAttributeParam(builder, param, attrValue);
            replaced = true;
        }
        if (replaced)
            paramsToRemove.add(param);
    }
    for (auto param : paramsToRemove)
        param->removeAndDeallocate();

    // Falling off the end of an anyhit shader accepts the candidate.
    rewriteVoidReturns(func, builder, [&](IRBuilder& b) { return b.getBoolValue(true); });

    fixUpFuncType(func, builder.getBoolType());

    context.rtEntries[func] = info;
}

/// Rewrite an intersection entry point into a Metal bounding-box
/// intersection function returning `slang_RTBBoxResult`. The user's code
/// reports hits through `ReportHit`, which the op lowering turns into the
/// accept/reject protocol; falling off the end reports no hit.
static void processIntersectionEntryPoint(
    MetalRayTracingLegalizationContext& context,
    IRFunc* func)
{
    ensureSharedTypes(context);

    IRBuilder builder(context.module);
    auto firstBlock = func->getFirstBlock();
    SLANG_ASSERT(firstBlock);

    auto floatType = builder.getBasicType(BaseType::Float);

    MetalRayTracingLegalizationContext::RTEntryInfo info;
    info.stage = Stage::Intersection;
    addIntersectionFunctionCommonParams(context, builder, func, info);
    auto minTParam =
        addSystemParam(builder, func, floatType, "slang_rtMinT", String("min_distance"));
    auto maxTParam =
        addSystemParam(builder, func, floatType, "slang_rtMaxT", String("max_distance"));
    addIntersectionFunctionContextParam(context, builder, func, info);
    info.isectMinT = minTParam;
    info.isectMaxT = maxTParam;
    info.readerOverrides[kIROp_MetalRTRayTMin] = minTParam;
    info.readerOverrides[kIROp_MetalRTRayTCurrent] = maxTParam;

    // Falling off the end of an intersection shader reports no hit.
    auto bboxResultType = context.bboxResultType;
    rewriteVoidReturns(
        func,
        builder,
        [&](IRBuilder& b)
        {
            IRInst* args[] = {
                b.getBoolValue(false),
                b.getFloatValue(b.getBasicType(BaseType::Float), 0.0)};
            return b.emitMakeStruct(bboxResultType, 2, args);
        });

    fixUpFuncType(func, bboxResultType);

    context.rtEntries[func] = info;
}

/// Diagnose references to globally bound shader parameters or mutable global
/// variables from a handler-stage entry point, including references made by
/// helper functions it calls. Visible and intersection functions cannot see
/// the kernel's resource bindings, and Metal has no module-scope mutable
/// state; until the `slang_RTGlobals` hoisting of section 4.3 is
/// implemented, such references cannot be honored and must be rejected
/// loudly rather than emitted as silently broken MSL. The check walks the
/// direct-call graph because after `inlineMetalRTOpsIntoEntryPoints` only
/// helpers *without* ray-tracing ops remain outlined, and those can still
/// touch globals (`introduceExplicitGlobalContext` would later thread a
/// kernel-context parameter through them into the handler, corrupting its
/// fixed ABI — that pass carries a backstop diagnostic for the same
/// condition).
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
/// fixed indices reserved for the ray-generation kernel's system parameters.
/// User bindings are laid out from index 0 upward, so a collision requires a
/// very large binding range or an explicit user-chosen register — either way
/// it must fail loudly, because the kernel would otherwise bind two
/// resources to one slot.
static void checkForReservedBufferIndexCollisions(
    MetalRayTracingLegalizationContext& context,
    int reservedBase)
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
        if (offset >= (UInt)reservedBase)
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

/// Pack the caller's payload (pointed to by `payloadPtr`) into the context's
/// payload blob ahead of a handler dispatch, returning the typed blob
/// address the caller passes to `emitPayloadBlobUnpack` after the dispatch.
static IRInst* emitPayloadBlobPack(
    MetalRayTracingLegalizationContext& context,
    IRBuilder& builder,
    IRInst* ctxPtr,
    IRInst* payloadPtr,
    IRInst* diagnosticInst)
{
    auto payloadType = (IRType*)as<IRPtrTypeBase>(payloadPtr->getDataType())->getValueType();
    auto typedBlobAddr = emitTypedPayloadBlobAddr(
        context,
        builder,
        ctxPtr,
        payloadType,
        AddressSpace::ThreadLocal,
        diagnosticInst);
    builder.emitStore(typedBlobAddr, builder.emitLoad(payloadPtr));
    return typedBlobAddr;
}

/// Unpack the context's payload blob back into the caller's payload after a
/// handler dispatch, so the caller observes the handler's writes.
static void emitPayloadBlobUnpack(IRBuilder& builder, IRInst* typedBlobAddr, IRInst* payloadPtr)
{
    builder.emitStore(payloadPtr, builder.emitLoad(typedBlobAddr));
}

/// Call the visible-function-table record `recordIndex` with the entry's
/// context and globals pointers — the dispatch primitive shared by the
/// TraceRay hit/miss paths (and, once phase P2 lands, CallShader).
static void emitHandlerTableCall(
    IRBuilder& builder,
    MetalRayTracingLegalizationContext::RTEntryInfo& entryInfo,
    IRInst* recordIndex)
{
    IRInst* callArgs[] = {
        entryInfo.handlerTable,
        recordIndex,
        entryInfo.ctxPtr,
        entryInfo.globalsPtr};
    builder.emitIntrinsicInst(builder.getVoidType(), kIROp_MetalRTHandlerCall, 4, callArgs);
}

/// Move `inst` and everything after it in its block to the end of `toBlock`,
/// preserving order. Used when an op lowering splits a block with
/// `emitIfElseWithBlocks`: the tail of the original block becomes the tail
/// of the merge block.
static void moveTailIntoBlock(IRInst* inst, IRBlock* toBlock)
{
    List<IRInst*> instsToMove;
    for (IRInst* cursor = inst; cursor; cursor = cursor->getNextInst())
        instsToMove.add(cursor);
    for (auto instToMove : instsToMove)
        instToMove->insertAtEnd(toBlock);
}

/// Replace `inst` (a terminator-like op such as the lowering of IgnoreHit)
/// with `return returnValue`, removing the now-unreachable remainder of its
/// block.
static void replaceInstWithReturn(IRBuilder& builder, IRInst* inst, IRInst* returnValue)
{
    builder.setInsertBefore(inst);
    builder.emitReturn(returnValue);
    List<IRInst*> deadInsts;
    for (IRInst* cursor = inst; cursor; cursor = cursor->getNextInst())
        deadInsts.add(cursor);
    for (Index i = deadInsts.getCount() - 1; i >= 0; i--)
        deadInsts[i]->removeAndDeallocate();
}

/// Lower a `metalRTTraceRay` op inside a ray-generation kernel into: stores
/// of the ray state into the context, payload packing, traversal through the
/// `metalRTIntersect` op (the `_slang_rtTrace` prelude helper, which runs
/// `intersector<>::intersect()` and fills the committed-hit context fields),
/// and the shader-binding-table dispatch of the closest-hit or miss shader:
///   hit:  record = hitBase + RayContributionToHitGroupIndex
///                + MultiplierForGeometryContributionToHitGroupIndex * ctx.geometryIndex
///                + instanceSbtOffsets[ctx.instanceIndex]
///   miss: record = missBase + MissShaderIndex
/// with the per-instance contribution supplied through the SBT buffer,
/// because Metal's instance descriptor offset only applies to intersection
/// function tables, not visible function tables.
static void lowerTraceRayOp(
    MetalRayTracingLegalizationContext& context,
    MetalRayTracingLegalizationContext::RTEntryInfo& entryInfo,
    IRMetalRTTraceRay* inst)
{
    IRBuilder builder(context.module);
    builder.setInsertBefore(inst);

    auto uintType = builder.getUIntType();
    auto ctxPtr = entryInfo.ctxPtr;

    storeContextField(builder, ctxPtr, context.originKey, inst->getOrigin());
    storeContextField(builder, ctxPtr, context.tMinKey, inst->getTMin());
    storeContextField(builder, ctxPtr, context.directionKey, inst->getDirection());
    storeContextField(builder, ctxPtr, context.tMaxKey, inst->getTMax());
    storeContextField(builder, ctxPtr, context.rayFlagsKey, inst->getRayFlags());

    auto payloadPtr = inst->getPayloadPtr();
    auto typedBlobAddr = emitPayloadBlobPack(context, builder, ctxPtr, payloadPtr, inst);

    IRInst* status = nullptr;
    if (entryInfo.isectTable)
    {
        IRInst* intersectArgs[] = {
            ctxPtr,
            inst->getAccelerationStructure(),
            inst->getInstanceInclusionMask(),
            inst->getRayFlags(),
            entryInfo.isectTable};
        status = builder.emitIntrinsicInst(uintType, kIROp_MetalRTIntersect, 5, intersectArgs);
    }
    else
    {
        IRInst* intersectArgs[] = {
            ctxPtr,
            inst->getAccelerationStructure(),
            inst->getInstanceInclusionMask(),
            inst->getRayFlags()};
        status = builder.emitIntrinsicInst(uintType, kIROp_MetalRTIntersect, 4, intersectArgs);
    }

    IRInst* neqArgs[] = {status, builder.getIntValue(uintType, kMetalRTCommittedNone)};
    auto isHit = builder.emitIntrinsicInst(builder.getBoolType(), kIROp_Neq, 2, neqArgs);

    IRBlock* hitBlock = nullptr;
    IRBlock* missBlock = nullptr;
    IRBlock* mergeBlock = nullptr;
    builder.emitIfElseWithBlocks(isHit, hitBlock, missBlock, mergeBlock);

    // Hit path: DXR hit-group record indexing against the committed-hit
    // state the trace helper stored into the context.
    builder.setInsertInto(hitBlock);
    {
        auto hitBase =
            builder.emitLoad(builder.emitFieldAddress(entryInfo.sbtPtr, context.hitBaseKey));
        auto offsetsPtr = builder.emitLoad(
            builder.emitFieldAddress(entryInfo.sbtPtr, context.instanceSbtOffsetsKey));
        auto geometryIndex =
            builder.emitLoad(builder.emitFieldAddress(ctxPtr, context.geometryIndexKey));
        auto instanceIndex =
            builder.emitLoad(builder.emitFieldAddress(ctxPtr, context.instanceIndexKey));
        auto instanceContribution =
            builder.emitLoad(builder.emitGetOffsetPtr(offsetsPtr, instanceIndex));

        auto recordIndex =
            builder.emitAdd(uintType, hitBase, inst->getRayContributionToHitGroupIndex());
        recordIndex = builder.emitAdd(
            uintType,
            recordIndex,
            builder
                .emitMul(uintType, inst->getMultiplierForGeometryContribution(), geometryIndex));
        recordIndex = builder.emitAdd(uintType, recordIndex, instanceContribution);

        // DXR RAY_FLAG_SKIP_CLOSEST_HIT_SHADER (0x08) suppresses only the
        // closest-hit dispatch; traversal and the committed-hit state are
        // unaffected, and the miss path still dispatches.
        IRInst* maskArgs[] = {inst->getRayFlags(), builder.getIntValue(uintType, 0x08)};
        auto skipBits = builder.emitIntrinsicInst(uintType, kIROp_BitAnd, 2, maskArgs);
        IRInst* eqlArgs[] = {skipBits, builder.getIntValue(uintType, 0)};
        auto shouldDispatch =
            builder.emitIntrinsicInst(builder.getBoolType(), kIROp_Eql, 2, eqlArgs);
        IRBlock* dispatchBlock = nullptr;
        IRBlock* skipBlock = nullptr;
        builder.emitIfWithBlocks(shouldDispatch, dispatchBlock, skipBlock);
        builder.setInsertInto(dispatchBlock);
        emitHandlerTableCall(builder, entryInfo, recordIndex);
        builder.emitBranch(skipBlock);
        builder.setInsertInto(skipBlock);
        builder.emitBranch(mergeBlock);
    }

    // Miss path.
    builder.setInsertInto(missBlock);
    {
        // In a miss shader DXR defines RayTCurrent() as the ray's TMax.
        storeContextField(builder, ctxPtr, context.hitTKey, inst->getTMax());
        auto missBase =
            builder.emitLoad(builder.emitFieldAddress(entryInfo.sbtPtr, context.missBaseKey));
        auto recordIndex = builder.emitAdd(uintType, missBase, inst->getMissShaderIndex());
        emitHandlerTableCall(builder, entryInfo, recordIndex);
        builder.emitBranch(mergeBlock);
    }

    // The rest of the original block continues after the dispatch; unpack
    // the payload there so the caller observes the handler's writes.
    moveTailIntoBlock(inst, mergeBlock);
    builder.setInsertBefore(inst);
    emitPayloadBlobUnpack(builder, typedBlobAddr, payloadPtr);
    inst->removeAndDeallocate();
}

/// Lower a `metalRTCallShader` op inside a ray-generation kernel into the
/// section 5.2 dispatch: pack the callable data into the context's payload
/// blob, call the record `callableBase + shaderIndex` through the visible
/// function table, and unpack the blob back.
static void lowerCallShaderOp(
    MetalRayTracingLegalizationContext& context,
    MetalRayTracingLegalizationContext::RTEntryInfo& entryInfo,
    IRMetalRTCallShader* inst)
{
    IRBuilder builder(context.module);
    builder.setInsertBefore(inst);

    auto uintType = builder.getUIntType();

    auto payloadPtr = inst->getPayloadPtr();
    auto typedBlobAddr = emitPayloadBlobPack(context, builder, entryInfo.ctxPtr, payloadPtr, inst);

    auto callableBase =
        builder.emitLoad(builder.emitFieldAddress(entryInfo.sbtPtr, context.callableBaseKey));
    auto recordIndex = builder.emitAdd(uintType, callableBase, inst->getShaderIndex());
    emitHandlerTableCall(builder, entryInfo, recordIndex);

    emitPayloadBlobUnpack(builder, typedBlobAddr, payloadPtr);
    inst->removeAndDeallocate();
}

/// Lower a `metalRTReportHit` op inside an intersection entry point into the
/// bounding-box accept protocol: when tHit lies inside the current search
/// interval, store the attributes and hit kind into the context and return
/// an accepting result carrying tHit; otherwise the op yields false and
/// execution continues. Storing only in-interval hits keeps the context's
/// committed attribute state consistent: Metal commits every accepted
/// in-interval hit (shrinking the interval), so the last write is always the
/// final committed hit.
static void lowerReportHitOp(
    MetalRayTracingLegalizationContext& context,
    MetalRayTracingLegalizationContext::RTEntryInfo& entryInfo,
    IRMetalRTReportHit* inst)
{
    IRBuilder builder(context.module);
    builder.setInsertBefore(inst);

    auto boolType = builder.getBoolType();
    auto hitT = inst->getHitT();

    IRInst* geqArgs[] = {hitT, entryInfo.isectMinT};
    auto aboveMin = builder.emitIntrinsicInst(boolType, kIROp_Geq, 2, geqArgs);
    IRInst* leqArgs[] = {hitT, entryInfo.isectMaxT};
    auto belowMax = builder.emitIntrinsicInst(boolType, kIROp_Leq, 2, leqArgs);
    IRInst* andArgs[] = {aboveMin, belowMax};
    auto inRange = builder.emitIntrinsicInst(boolType, kIROp_And, 2, andArgs);

    IRBlock* acceptBlock = nullptr;
    IRBlock* rejectBlock = nullptr;
    IRBlock* mergeBlock = nullptr;
    builder.emitIfElseWithBlocks(inRange, acceptBlock, rejectBlock, mergeBlock);

    builder.setInsertInto(acceptBlock);
    {
        auto attrValue = inst->getAttributes();
        auto typedAttrAddr = emitTypedAttributesBlobAddr(
            context,
            builder,
            entryInfo.ctxPtr,
            attrValue->getDataType(),
            AddressSpace::MetalRayData,
            inst);
        builder.emitStore(typedAttrAddr, attrValue);
        storeContextField(builder, entryInfo.ctxPtr, context.hitKindKey, inst->getHitKind());
        IRInst* resultArgs[] = {builder.getBoolValue(true), hitT};
        auto acceptResult = builder.emitMakeStruct(context.bboxResultType, 2, resultArgs);
        builder.emitReturn(acceptResult);
    }

    builder.setInsertInto(rejectBlock);
    builder.emitBranch(mergeBlock);

    moveTailIntoBlock(inst, mergeBlock);
    inst->replaceUsesWith(builder.getBoolValue(false));
    inst->removeAndDeallocate();
}

/// Lower every transient `metalRT*` op inside a rewritten ray-tracing entry
/// point: system-value readers become parameter references or loads of
/// context fields, `metalRTTraceRay` becomes the traversal-plus-dispatch
/// sequence, and the ReportHit/IgnoreHit/AcceptHitAndEndSearch family
/// becomes the Metal intersection-function accept/reject protocol.
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
        IRBuilder builder(context.module);
        switch (inst->getOp())
        {
        case kIROp_MetalRTTraceRay:
        case kIROp_MetalRTCallShader:
            {
                // Dispatch works wherever the table/SBT state is reachable:
                // the ray-generation kernel (system parameters) and the
                // miss/closest-hit/callable handlers (via the globals
                // argument buffer, section 5.3). The capability system keeps
                // these intrinsics out of the remaining stages, so a missing
                // table here is out-of-contract.
                if (!entryInfo.handlerTable)
                {
                    context.sink->diagnose(
                        Diagnostics::MetalRaytracingIntrinsicOutsideEntryPoint{
                            .location = getDiagnosticPos(inst)});
                    inst->removeAndDeallocate();
                    continue;
                }
                if (auto callShader = as<IRMetalRTCallShader>(inst))
                    lowerCallShaderOp(context, entryInfo, callShader);
                else
                    lowerTraceRayOp(context, entryInfo, cast<IRMetalRTTraceRay>(inst));
            }
            break;
        case kIROp_MetalRTReportHit:
            {
                if (entryInfo.stage != Stage::Intersection)
                {
                    context.sink->diagnose(
                        Diagnostics::MetalRaytracingIntrinsicOutsideEntryPoint{
                            .location = getDiagnosticPos(inst)});
                    inst->replaceUsesWith(builder.getBoolValue(false));
                    inst->removeAndDeallocate();
                    continue;
                }
                lowerReportHitOp(context, entryInfo, cast<IRMetalRTReportHit>(inst));
            }
            break;
        case kIROp_MetalRTIgnoreHit:
        case kIROp_MetalRTAcceptHitAndEndSearch:
            {
                if (entryInfo.stage != Stage::AnyHit)
                {
                    context.sink->diagnose(
                        Diagnostics::MetalRaytracingIntrinsicOutsideEntryPoint{
                            .location = getDiagnosticPos(inst)});
                    inst->removeAndDeallocate();
                    continue;
                }
                // IgnoreHit rejects the candidate. AcceptHitAndEndSearch
                // accepts it; the search only actually ends when the ray was
                // traced with RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
                // (`accept_any_intersection` in the intersector), see the
                // design doc's section 4.5.
                bool accepts = inst->getOp() == kIROp_MetalRTAcceptHitAndEndSearch;
                replaceInstWithReturn(builder, inst, builder.getBoolValue(accepts));
            }
            break;
        default:
            {
                IRInst* overrideValue = nullptr;
                if (entryInfo.readerOverrides.tryGetValue((int)inst->getOp(), overrideValue))
                {
                    inst->replaceUsesWith(overrideValue);
                    inst->removeAndDeallocate();
                    continue;
                }
                auto key = getContextKeyForReaderOp(context, inst->getOp());
                SLANG_ASSERT(key);
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
                    context.sink->diagnose(
                        Diagnostics::MetalRaytracingIntrinsicOutsideEntryPoint{
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
    List<EntryPointInfo> anyHitEntryPoints;
    List<EntryPointInfo> intersectionEntryPoints;
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
        // A callable shader has the same shape as a miss shader from the
        // execution model's point of view: one inout data parameter bridged
        // through the context's payload blob, dispatched through the
        // visible function table (from its own SBT region).
        case Stage::Callable:
            anyRTEntryPoints = true;
            handlerEntryPoints.add(entryPoint);
            break;
        case Stage::AnyHit:
            anyRTEntryPoints = true;
            anyHitEntryPoints.add(entryPoint);
            break;
        case Stage::Intersection:
            anyRTEntryPoints = true;
            intersectionEntryPoints.add(entryPoint);
            break;
        default:
            break;
        }
    }
    if (!anyRTEntryPoints)
        return;

    context.hasIntersectionStages =
        anyHitEntryPoints.getCount() != 0 || intersectionEntryPoints.getCount() != 0;

    // The raygen kernel's system parameters live at fixed buffer indices;
    // user bindings must stay below them. The intersection function table
    // index is only reserved when such stages are present.
    if (rayGenEntryPoints.getCount() != 0)
    {
        checkForReservedBufferIndexCollisions(
            context,
            context.hasIntersectionStages ? kMetalRTIsectBufferIndex
                                          : kMetalRTGlobalsBufferIndex);
    }

    inlineMetalRTOpsIntoEntryPoints(module);

    // With every handler-side global reference now physically inside its
    // entry point, decide the layout of the user-resource tail of
    // `slang_RTGlobals` before any entry is rewritten.
    hoistHandlerGlobals(context, handlerEntryPoints);

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
    for (auto& entryPoint : anyHitEntryPoints)
    {
        processAnyHitEntryPoint(context, entryPoint.entryPointFunc);
        checkForGlobalStateInHandler(context, entryPoint.entryPointFunc);
    }
    for (auto& entryPoint : intersectionEntryPoints)
    {
        processIntersectionEntryPoint(context, entryPoint.entryPointFunc);
        checkForGlobalStateInHandler(context, entryPoint.entryPointFunc);
    }

    for (auto& [func, info] : context.rtEntries)
        lowerMetalRTOpsInFunc(context, func, info);

    diagnoseStrayMetalRTOps(context);
}

} // namespace Slang
