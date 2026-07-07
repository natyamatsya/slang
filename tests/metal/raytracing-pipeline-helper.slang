//TEST:SIMPLE(filecheck=METAL): -target metal

// TraceRay called from a user helper function must end up inside the
// ray-generation kernel, where the ray-tracing system parameters are in
// scope (legalizeMetalRayTracing inlines helper functions that contain
// ray-tracing ops into their entry points). Two traces with two different
// payload types share the one 64-byte context payload blob, each through
// its own typed view.

uniform RaytracingAccelerationStructure scene;
RWStructuredBuffer<float4> outputBuffer;

struct ColorPayload
{
    float4 color;
};

struct ShadowPayload
{
    float visibility;
};

float4 shootColorRay(RayDesc ray)
{
    ColorPayload payload;
    payload.color = float4(0, 0, 0, 0);
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload);
    return payload.color;
}

// Everything lands in the kernel; both payload types are bridged through
// the same context blob with their own reinterpretation:
// METAL: {{\[\[}}kernel{{\]\]}} void rayGenMain(
// METAL-DAG: (ColorPayload{{[^)]*}} thread*)(&(&slang_rtCtx{{[^)]*}})->payload
// METAL-DAG: (ShadowPayload{{[^)]*}} thread*)(&(&slang_rtCtx{{[^)]*}})->payload
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

    float4 color = shootColorRay(ray);

    ShadowPayload shadow;
    shadow.visibility = 0.0;
    TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff, 1, 1, 1, ray, shadow);

    outputBuffer[idx.y * dim.x + idx.x] = color * shadow.visibility;
}

// METAL-DAG: {{\[\[}}visible{{\]\]}} void colorMiss(
[shader("miss")]
void colorMiss(inout ColorPayload payload)
{
    payload.color = float4(0.1, 0.2, 0.3, 1.0);
}

// METAL-DAG: {{\[\[}}visible{{\]\]}} void shadowMiss(
[shader("miss")]
void shadowMiss(inout ShadowPayload payload)
{
    payload.visibility = 1.0;
}

// METAL-DAG: {{\[\[}}visible{{\]\]}} void colorHit(
[shader("closesthit")]
void colorHit(inout ColorPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.color = float4(attr.barycentrics, 0, 1);
}

// METAL-DAG: {{\[\[}}visible{{\]\]}} void shadowHit(
[shader("closesthit")]
void shadowHit(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.visibility = 0.0;
}
