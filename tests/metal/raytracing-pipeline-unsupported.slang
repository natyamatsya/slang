//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target metal

// Features beyond the implemented phases of docs/design/metal-raytracing.md
// (module-scope mutable state in handler stages, global resource access
// from anyhit/intersection functions, out-of-contract attribute types) must
// fail loudly until implemented.

StructuredBuffer<float4> missColors;

struct RayPayload
{
    float4 color;
};

// Mutable module-scope state cannot be hoisted into the slang_RTGlobals
// argument buffer:
static float gAccum = 0.0;

//CHECK-DAG: 56113
[shader("miss")]
void missGlobalVar(inout RayPayload payload)
{
    gAccum += 1.0;
    payload.color = float4(gAccum, 0, 0, 1);
}

// Global resource access from anyhit/intersection functions is not wired
// up yet (they have no slang_RTGlobals parameter until the
// intersection-table buffer binding lands):
//CHECK-DAG: 56113
[shader("anyhit")]
void anyHitGlobal(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.color = missColors[0] + float4(attr.barycentrics, 0, 0);
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
