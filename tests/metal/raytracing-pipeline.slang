//TEST:SIMPLE(filecheck=METAL): -target metal

// Test the P0 Metal ray-tracing pipeline lowering of
// docs/design/metal-raytracing.md: a ray-generation shader becomes the
// compute kernel with the implicit system parameters, TraceRay lowers to an
// intersection-query traversal plus a visible-function-table dispatch with
// DXR shader-binding-table indexing, and miss/closest-hit shaders become
// [[visible]] functions with the uniform handler signature that bridge the
// payload through the context blob.

uniform RaytracingAccelerationStructure scene;
RWStructuredBuffer<float4> outputBuffer;

struct RayPayload
{
    float4 color;
};

// The ray-generation shader is the [[kernel]], with the visible function
// table, software SBT, and globals argument buffer as implicit trailing
// parameters at the documented fixed binding indices, and the launch state
// coming from the compute dispatch coordinates:
// METAL: {{\[\[}}kernel{{\]\]}} void rayGenMain(
// METAL-SAME: visible_function_table<void(slang_RTContext{{[^,]*}} thread*, slang_RTGlobals{{[^,]*}} device*)> slang_rtHandlers{{[^ ]*}} {{\[\[}}buffer(30){{\]\]}}
// METAL-SAME: slang_RTSbt{{[^ ]*}} constant* slang_rtSbt{{[^ ]*}} {{\[\[}}buffer(29){{\]\]}}
// METAL-SAME: slang_RTGlobals{{[^ ]*}} device* slang_rtGlobals{{[^ ]*}} {{\[\[}}buffer(28){{\]\]}}
// METAL-SAME: {{\[\[}}thread_position_in_grid{{\]\]}}
// METAL-SAME: {{\[\[}}threads_per_grid{{\]\]}}
[shader("raygeneration")]
void rayGenMain()
{
    uint3 idx = DispatchRaysIndex();
    uint3 dim = DispatchRaysDimensions();

    RayDesc ray;
    ray.Origin = float3(idx.x, idx.y, -1.0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.001;
    ray.TMax = 1000.0;

    RayPayload payload;
    payload.color = float4(0, 0, 0, 0);

    // The payload is packed into the context blob, then the traversal half
    // runs through the _slang_rtTrace prelude helper (intersector<>-based;
    // this triangles-only program links no intersection function table, so
    // the no-table overload is called):
    // METAL: (RayPayload{{[^)]*}} thread*)(&(&slang_rtCtx{{[^)]*}})->payload)
    // METAL: _slang_rtTrace(&slang_rtCtx{{[^,]*}},
    //
    // The hit dispatch computes the DXR hit-group record index against the
    // committed-hit state the helper stored in the context
    // (hitBase + rayContribution + multiplier * ctx.geometryIndex
    //  + instanceSbtOffsets[ctx.instanceIndex]; the 0-contribution and
    // 1-multiplier of this call fold away) and calls through the table:
    // METAL: slang_rtHandlers{{[^[]*}}[{{[^]]*}}->hitBase{{[^]]*}}geometryIndex{{[^]]*}}instanceSbtOffsets{{[^]]*}}](&slang_rtCtx{{[^,]*}}, slang_rtGlobals{{[^)]*}})
    //
    // The miss dispatch stores the ray's TMax into the committed-distance
    // field (DXR: RayTCurrent() in a miss shader equals RayTMax) and indexes
    // from missBase:
    // METAL: ->hitT{{[^=]*}}= {{.*}}TMax
    // METAL: slang_rtHandlers{{[^[]*}}[{{[^]]*}}->missBase{{[^]]*}}](&slang_rtCtx{{[^,]*}}, slang_rtGlobals{{[^)]*}})
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload);

    outputBuffer[idx.y * dim.x + idx.x] = payload.color;
}

// The miss shader becomes a [[visible]] function with the uniform handler
// signature, unpacking the payload from the context blob on entry and
// packing it back before returning. RayTCurrent() reads the hitT context
// field, which the miss dispatch filled with the ray's TMax:
// METAL: {{\[\[}}visible{{\]\]}} void missMain(slang_RTContext{{[^,]*}} thread* slang_rtCtx{{[^,]*}}, slang_RTGlobals{{[^,]*}} device* slang_rtGlobals{{[^)]*}})
// METAL: (RayPayload{{[^)]*}} thread*)(&slang_rtCtx{{[^)]*}}->payload
// METAL: ->hitT
[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.color = float4(0.1, 0.2, 0.3, RayTCurrent());
}

// So does the closest-hit shader; its intersection attributes are rebuilt
// from the committed barycentrics in the context:
// METAL: {{\[\[}}visible{{\]\]}} void closestHitMain(slang_RTContext{{[^,]*}} thread* slang_rtCtx{{[^,]*}}, slang_RTGlobals{{[^,]*}} device* slang_rtGlobals{{[^)]*}})
// METAL: triBarycentrics
[shader("closesthit")]
void closestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.color = float4(attr.barycentrics.x, attr.barycentrics.y, RayTCurrent(), 1.0);
}
