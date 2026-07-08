//TEST:SIMPLE(filecheck=METAL): -target metal

// Test the P3 Metal lowering (docs/design/metal-raytracing.md sections 4.3,
// 5.3 and 10): handler stages can access hoisted global resources and can
// themselves TraceRay/CallShader through the dispatch state carried in the
// slang_RTGlobals argument buffer, and system-value reads are snapshotted at
// handler entry so a nested trace cannot corrupt them.

uniform RaytracingAccelerationStructure scene;
RWStructuredBuffer<float4> outputBuffer;
StructuredBuffer<float4> skyColors;

struct RayPayload
{
    float4 color;
    uint depth;
};

struct ShadeData
{
    float3 value;
    float pad;
};

// slang_RTGlobals carries the fixed dispatch-state header followed by the
// hoisted user resources, in declaration order:
// METAL: struct slang_RTGlobals{{$}}
// METAL-NEXT: {
// METAL-NEXT: visible_function_table<void(slang_RTContext thread*, slang_RTGlobals device*)> handlers;
// METAL-NEXT: {{.*}}intersection_function_table<{{.*}}> isect;
// METAL-NEXT: slang_RTSbt sbt;
// METAL-NEXT: {{.*}}acceleration_structure<{{.*}}> scene;
// METAL-NEXT: {{.*}} skyColors;
// METAL-NEXT: };
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
    payload.depth = 0;
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload);
    outputBuffer[idx.x] = payload.color;
}

// A miss shader reading a hoisted global and calling a callable: the
// resource comes from the globals argument buffer, and the dispatch goes
// through the handlers table loaded from it:
// METAL-DAG: {{\[\[}}visible{{\]\]}} void missMain(
// METAL-DAG: slang_rtGlobals{{[^-]*}}->skyColors
// METAL-DAG: ->handlers
// METAL-DAG: ->callableBase
[shader("miss")]
void missMain(inout RayPayload payload)
{
    ShadeData shade;
    shade.value = skyColors[0].rgb;
    shade.pad = 0;
    CallShader(0, shade);
    payload.color = float4(shade.value, RayTCurrent());
}

[shader("callable")]
void skyFilter(inout ShadeData data)
{
    data.value = data.value * 0.5;
}

// A closest-hit shader tracing a nested shadow ray: the scene comes from
// the globals argument buffer and traversal runs inside the visible
// function; the committed-hit reads (RayTCurrent, PrimitiveIndex, ...)
// use entry snapshots taken before the nested trace can overwrite the
// context:
// METAL-DAG: {{\[\[}}visible{{\]\]}} void closestHitMain(
// METAL-DAG: slang_rtGlobals{{[^-]*}}->scene
// METAL-DAG: _slang_rtTrace(slang_rtCtx
[shader("closesthit")]
void closestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    float firstHitT = RayTCurrent();
    uint firstPrim = PrimitiveIndex();
    payload.color = float4(attr.barycentrics, 0, 1);
    if (payload.depth < 1)
    {
        RayDesc shadow;
        shadow.Origin = WorldRayOrigin() + WorldRayDirection() * firstHitT;
        shadow.Direction = float3(0, 1, 0);
        shadow.TMin = 0.01;
        shadow.TMax = 100.0;
        RayPayload nested;
        nested.color = float4(1, 1, 1, 1);
        nested.depth = payload.depth + 1;
        TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff, 0, 1, 0, shadow, nested);
        payload.color *= nested.color.x;
        payload.color.z = firstHitT + float(firstPrim) + float(PrimitiveIndex());
    }
}
