//TEST:SIMPLE(filecheck=METAL): -target metal -warnings-disable 39029
//TEST:SIMPLE(filecheck=SLOTS): -target metal -warnings-disable 39029 -metal-rt-globals-slots 4

// Test the intersection-table globals path (runtime contract section 4):
// an anyhit or intersection function that references a globally bound
// resource gains one trailing `slang_RTGlobals device* [[buffer(0)]]`
// parameter — intersection-table buffer slot 0, a distinct index space
// from the kernel's buffer table, bound by the runtime via setBuffer —
// and reads the resource from the same hoisted tail as the other handler
// stages. Functions that reference nothing declare nothing.

uniform RaytracingAccelerationStructure scene : register(t3);
RWStructuredBuffer<float4> outputBuffer : register(u0);
StructuredBuffer<float4> opacity : register(t1);
StructuredBuffer<float4> sphereData : register(t2);

struct RayPayload { float4 color; };

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

// The globals-referencing anyhit gains the trailing buffer(0) parameter
// and reads through it; `opacity` is hoisted into the shared tail:
// METAL-DAG: bool anyHitMain({{.*}}slang_RTGlobals device* slang_rtGlobals{{[^ ]*}} {{\[\[}}buffer(0){{\]\]}})
// METAL-DAG: slang_rtGlobals{{[^-]*}}->opacity
// METAL-DAG: packed_float4 device* opacity;
[shader("anyhit")]
void anyHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    if (opacity[PrimitiveIndex()].x < 0.5)
        IgnoreHit();
}

// A non-referencing intersection-stage function declares no globals
// parameter (per-use; slot 0 is bound unconditionally by the runtime):
// METAL-DAG: bool anyHitClean({{[^)]*}}payload{{\]\]}}){{$}}
[shader("anyhit")]
void anyHitClean(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    if (attr.barycentrics.x < 0.25)
        IgnoreHit();
}

// An intersection function referencing globals gains the same trailing
// parameter and reads from the same tail:
// METAL-DAG: slang_RTBBoxResult isectMain({{.*}}slang_RTGlobals device* slang_rtGlobals{{[^ ]*}} {{\[\[}}buffer(0){{\]\]}})
// METAL-DAG: slang_rtGlobals{{[^-]*}}->sphereData
//
// Under the slot-addressed layout the same functions read from
// register-addressed slots (opacity at t1, sphereData at t2), and the
// per-use trailing parameter is unchanged:
// SLOTS-DAG: // slang-metal-rt-abi:{{.*}}"slots":4,
// SLOTS-DAG: bool anyHitMain({{.*}}slang_RTGlobals device* slang_rtGlobals{{[^ ]*}} {{\[\[}}buffer(0){{\]\]}})
// SLOTS-DAG: ulong slang_rtSlot0;
// SLOTS-DAG: packed_float4 device* opacity;
// SLOTS-DAG: packed_float4 device* sphereData;
// SLOTS-DAG: ulong slang_rtSlot3;
struct SphereAttrs
{
    float3 normal;
    float pad;
};

[shader("intersection")]
void isectMain()
{
    float4 sphere = sphereData[GeometryIndex()];
    float3 toCenter = WorldRayOrigin() - sphere.xyz;
    float b = dot(toCenter, WorldRayDirection());
    float disc = b * b - (dot(toCenter, toCenter) - sphere.w * sphere.w);
    if (disc >= 0)
    {
        SphereAttrs attrs;
        attrs.normal = float3(0, 0, 1);
        attrs.pad = 0;
        ReportHit(-b - sqrt(disc), 7, attrs);
    }
}

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.color = float4(0.1, 0.2, 0.3, 1.0);
}

[shader("closesthit")]
void chitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.color = float4(attr.barycentrics, 0, 1) + opacity[0];
}
