//TEST:SIMPLE(filecheck=METAL): -target metal

// Test the P1 Metal lowering of intersection shaders and procedural-hit
// attributes (docs/design/metal-raytracing.md sections 4.5 and 10): an
// intersection shader becomes a Metal bounding-box intersection function
// returning the accept/distance result struct, ReportHit becomes the
// range-guarded accept protocol writing the attributes into the context
// blob, and a closest-hit shader with a user attribute struct reads the
// blob back.

uniform RaytracingAccelerationStructure scene;
RWStructuredBuffer<float4> outputBuffer;

struct RayPayload
{
    float4 color;
};

struct SphereAttrs
{
    float3 normal;
    float shifted;
};

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

// The emitted functions may appear in any order, so the per-function
// fragments are checked as DAG entries.
//
// The result struct carries Metal's fixed accept protocol:
// METAL-DAG: struct slang_RTBBoxResult
// METAL-DAG: bool accept {{\[\[}}accept_intersection{{\]\]}}
// METAL-DAG: float distance {{\[\[}}distance{{\]\]}}
//
// The intersection shader becomes a bounding-box intersection function with
// the search interval and candidate state as tagged parameters and the
// context as the ray_data payload reference:
// METAL-DAG: {{\[\[}}intersection(bounding_box, raytracing::triangle_data, raytracing::instancing){{\]\]}} slang_RTBBoxResult isectMain({{[^)]*}}min_distance{{[^)]*}}max_distance{{[^)]*}}ray_data slang_RTContext& slang_rtCtx{{[^ ]*}} {{\[\[}}payload{{\]\]}})
//
// ReportHit writes the attributes into the context's attribute blob (as a
// ray_data reinterpretation) and the hit kind into the context:
// METAL-DAG: (SphereAttrs{{[^)]*}} ray_data*)(&(&slang_rtCtx{{[^)]*}})->attributes)
// METAL-DAG: ->hitKind
[shader("intersection")]
void isectMain()
{
    float3 ro = WorldRayOrigin();
    float3 rd = WorldRayDirection();
    float b = dot(ro, rd);
    float c = dot(ro, ro) - 1.0;
    float disc = b * b - c;
    if (disc >= 0)
    {
        float t = -b - sqrt(disc);
        SphereAttrs attrs;
        attrs.normal = normalize(ro + t * rd);
        attrs.shifted = float(PrimitiveIndex());
        ReportHit(t, 7, attrs);
    }
}

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.color = float4(0.1, 0.2, 0.3, 1.0);
}

// The closest-hit shader reconstructs the user attribute struct from the
// context's attribute blob (not from the triangle barycentrics field):
// METAL-DAG: {{\[\[}}visible{{\]\]}} void closestHitMain(slang_RTContext{{[^,]*}} thread* slang_rtCtx{{[^,]*}}, slang_RTGlobals{{[^,]*}} device* slang_rtGlobals{{[^)]*}})
// METAL-DAG: (SphereAttrs{{[^)]*}} thread*)(&slang_rtCtx{{[^)]*}}->attributes)
[shader("closesthit")]
void closestHitMain(inout RayPayload payload, in SphereAttrs attrs)
{
    payload.color = float4(attrs.normal, HitKind() == 7 ? 1.0 : 0.0);
}
