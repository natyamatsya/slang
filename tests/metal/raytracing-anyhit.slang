//TEST:SIMPLE(filecheck=METAL): -target metal

// Test the P1 Metal lowering of anyhit shaders
// (docs/design/metal-raytracing.md sections 4.5 and 10): an anyhit shader
// becomes a Metal triangle intersection function that accepts or rejects
// the fixed-function triangle candidate, and a program that links one gets
// the `slang_rtIsect` intersection function table threaded into traversal.

uniform RaytracingAccelerationStructure scene;
RWStructuredBuffer<float4> outputBuffer;

struct RayPayload
{
    float4 color;
    uint touchedCount;
};

// The kernel gains the intersection function table at its reserved binding,
// and traversal passes it to the trace helper:
// METAL: {{\[\[}}kernel{{\]\]}} void rayGenMain(
// METAL-SAME: intersection_function_table<raytracing::triangle_data, raytracing::instancing> slang_rtIsect{{[^ ]*}} {{\[\[}}buffer(27){{\]\]}}
// METAL: _slang_rtTrace(&slang_rtCtx{{[^,]*}}, {{[^,]*}}, slang_rtIsect{{[^,]*}},
[shader("raygeneration")]
void rayGenMain()
{
    uint3 idx = DispatchRaysIndex();
    RayDesc ray;
    ray.Origin = float3(idx.x, idx.y, -1.0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.001;
    ray.TMax = 1000.0;
    RayPayload payload;
    payload.color = float4(0, 0, 0, 0);
    payload.touchedCount = 0;
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload);
    outputBuffer[idx.x] = payload.color + float(payload.touchedCount);
}

// The anyhit shader becomes a bool-returning triangle intersection function
// with the candidate state as tagged parameters and the context as the
// ray_data payload reference:
// METAL: {{\[\[}}intersection(triangle, raytracing::triangle_data, raytracing::instancing){{\]\]}} bool anyHitMain(
// METAL-SAME: {{\[\[}}distance{{\]\]}}
// METAL-SAME: {{\[\[}}barycentric_coord{{\]\]}}
// METAL-SAME: {{\[\[}}front_facing{{\]\]}}
// METAL-SAME: ray_data slang_RTContext& slang_rtCtx{{[^ ]*}} {{\[\[}}payload{{\]\]}}
//
// Payload access goes directly through the context blob (DXR: payload
// writes persist even for ignored hits), IgnoreHit rejects, and reaching
// the end of the shader accepts:
// METAL: (RayPayload{{[^)]*}} ray_data*)
// METAL: return false;
// METAL: return true;
[shader("anyhit")]
void anyHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.touchedCount += 1;
    if (attr.barycentrics.x < 0.5 && RayTCurrent() > RayTMin())
    {
        IgnoreHit();
    }
}

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.color = float4(0.1, 0.2, 0.3, 1.0);
}

[shader("closesthit")]
void closestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.color = float4(attr.barycentrics, 0, 1.0);
}
