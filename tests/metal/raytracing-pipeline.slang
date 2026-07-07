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

    // The traversal half reuses the proven intersection-query lowering:
    // METAL: intersection_query<raytracing::triangle_data, raytracing::instancing>
    //
    // The hit dispatch computes the DXR hit-group record index
    // (hitBase + rayContribution + multiplier * geometryIndex
    //  + instanceSbtOffsets[instanceIndex]; the 0-contribution and
    // 1-multiplier of this call fold away), packs the payload into the
    // context blob, and calls through the table:
    // METAL: get_committed_geometry_id
    //
    // Fixed-function triangle hit kinds: 254 for front faces, 255 for back
    // faces (MSL select returns its second operand when the condition is
    // true):
    // METAL: select(255U, 254U,
    // METAL: ->hitBase
    // METAL-SAME: instanceSbtOffsets
    // METAL: (RayPayload{{[^)]*}} thread*)(&(&slang_rtCtx{{[^)]*}})->payload
    // METAL: slang_rtHandlers{{[^[]*}}[{{[^]]*}}](&slang_rtCtx{{[^,]*}}, slang_rtGlobals{{[^)]*}})
    //
    // The miss dispatch stores the ray's TMax into the committed-distance
    // field (DXR: RayTCurrent() in a miss shader equals RayTMax) and indexes
    // from missBase:
    // METAL: ->hitT{{[^=]*}}= {{.*}}TMax
    // METAL: ->missBase
    // METAL: slang_rtHandlers{{[^[]*}}[{{[^]]*}}](&slang_rtCtx{{[^,]*}}, slang_rtGlobals{{[^)]*}})
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
