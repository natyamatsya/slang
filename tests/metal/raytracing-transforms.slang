//TEST:SIMPLE(filecheck=METAL): -target metal

// Test the world_space_data phase (docs/design/metal-raytracing.md sections
// 6 and 10): instance transforms and the object-space ray are available in
// every hit-group stage — as candidate parameter tags inside
// anyhit/intersection functions, and as committed context fields (filled by
// the trace helper) in closest-hit shaders.
//
// METAL-DAG: {{\[\[}}object_to_world_transform{{\]\]}}
// METAL-DAG: {{\[\[}}world_to_object_transform{{\]\]}}
// METAL-DAG: ->objectToWorld = hit.object_to_world_transform;
// METAL-DAG: ->objectRayOrigin = ctx->worldToObject * float4(ctx->origin, 1.0);
// METAL-DAG: slang_rtCtx{{[^-]*}}->objectToWorld
// METAL-DAG: slang_rtCtx{{[^-]*}}->worldToObject
uniform RaytracingAccelerationStructure scene;
RWStructuredBuffer<float4> outputBuffer;
struct RayPayload { float4 color; };
struct SphereAttrs { float3 objNormal; float pad; };

[shader("raygeneration")]
void rayGenMain()
{
    uint3 idx = DispatchRaysIndex();
    RayDesc ray;
    ray.Origin = float3(idx.x, idx.y, -1.0); ray.Direction = float3(0,0,1);
    ray.TMin = 0.001; ray.TMax = 1000.0;
    RayPayload p; p.color = 0;
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, p);
    outputBuffer[idx.x] = p.color;
}

[shader("intersection")]
void isectMain()
{
    // Object-space ray from candidate params; sphere at object origin.
    float3 ro = ObjectRayOrigin();
    float3 rd = ObjectRayDirection();
    float b = dot(ro, rd);
    float disc = b*b - (dot(ro,ro) - 1.0);
    if (disc >= 0)
    {
        SphereAttrs a;
        a.objNormal = normalize(ro + (-b - sqrt(disc)) * rd);
        a.pad = ObjectToWorld4x3()[3].x;   // translation column row? touch the matrix
        ReportHit(-b - sqrt(disc), 7, a);
    }
}

[shader("anyhit")]
void anyHitMain(inout RayPayload p, in BuiltInTriangleIntersectionAttributes a)
{
    if (ObjectRayOrigin().y < 0) IgnoreHit();
}

[shader("miss")]
void missMain(inout RayPayload p) { p.color = float4(0.1,0.1,0.1,1); }

[shader("closesthit")]
void chitMain(inout RayPayload p, in SphereAttrs a)
{
    // Committed transforms from context; world normal from object normal.
    float3x4 o2w = ObjectToWorld3x4();
    float4x3 w2o = WorldToObject4x3();
    float3 wn = normalize(mul(o2w, float4(a.objNormal, 0)) + ObjectRayDirection() * 0 + w2o[0] * 0);
    p.color = float4(wn, 1);
}
