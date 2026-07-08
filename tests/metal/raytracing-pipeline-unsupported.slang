//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target metal

// The P0 scope of docs/design/metal-raytracing.md supports raygeneration,
// miss, and closesthit only. Later phases (callable shaders, TraceRay
// recursion from a closest-hit shader, global resource access from handler
// stages via slang_RTGlobals) must fail loudly until implemented.

uniform RaytracingAccelerationStructure scene;
StructuredBuffer<float4> missColors;

struct RayPayload
{
    float4 color;
};

struct CallableData
{
    float4 value;
};

// Callable shaders arrive in phase P2:
//CHECK-DAG: 56110
[shader("callable")]
void callableMain(inout CallableData data)
{
    data.value = float4(1, 2, 3, 4);
}

// Global resource access from a handler stage needs the slang_RTGlobals
// hoisting of section 4.3, which is not implemented yet:
//CHECK-DAG: 56113
[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.color = missColors[0];
}

// The same restriction must hold when the global is reached *transitively*
// through a helper function: the helper would otherwise silently gain a
// kernel-context parameter and corrupt the fixed visible-function ABI of
// the handler that calls it.
float4 fetchMissColor()
{
    return missColors[1];
}

//CHECK-DAG: 56113
[shader("miss")]
void missTransitive(inout RayPayload payload)
{
    payload.color = fetchMissColor();
}

// Intersection attributes must be a struct (the built-in triangle shape or
// a user-defined struct for procedural hits):
//CHECK-DAG: 56114
[shader("closesthit")]
void closestHitBadAttrs(inout RayPayload payload, in float2 attrs)
{
    payload.color = float4(attrs, 0.0, 1.0);
}

// ... and must fit the 32-byte attribute blob of the context:
struct HugeAttrs
{
    float4 a;
    float4 b;
    float4 c; // 48 bytes > 32
};

//CHECK-DAG: 56117
[shader("closesthit")]
void closestHitHugeAttrs(inout RayPayload payload, in HugeAttrs attrs)
{
    payload.color = attrs.a;
}

// TraceRay from a closest-hit shader (recursion) arrives in phase P3:
//CHECK-DAG: 56111
[shader("closesthit")]
void closestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    RayDesc ray;
    ray.Origin = float3(0, 0, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.001;
    ray.TMax = 1000.0;
    RayPayload nested;
    nested.color = float4(0, 0, 0, 0);
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, nested);
    payload.color = nested.color + float4(attr.barycentrics, 0, 1);
}
