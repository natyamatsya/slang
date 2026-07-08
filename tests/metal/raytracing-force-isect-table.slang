//TEST:SIMPLE(filecheck=METAL): -target metal -metal-rt-force-isect-table

// -metal-rt-force-isect-table binds the slang_rtIsect intersection function
// table (and threads it into traversal) even though this module compiles no
// anyhit/intersection entry points, so a separately compiled ray-generation
// module can link against pipelines that use intersection functions.

uniform RaytracingAccelerationStructure scene;
RWStructuredBuffer<float4> outputBuffer;

struct RayPayload { float4 color; };

// METAL: {{\[\[}}kernel{{\]\]}} void rayGenMain(
// METAL-SAME: slang_rtIsect{{[^ ]*}} {{\[\[}}buffer(27){{\]\]}}
// METAL: _slang_rtTrace(&slang_rtCtx{{[^,]*}}, {{[^,]*}}, slang_rtIsect
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
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload);
    outputBuffer[idx.x] = payload.color;
}

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.color = float4(0.1, 0.2, 0.3, 1.0);
}
