//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target metal

// Out-of-contract input (module-scope mutable state in handler stages,
// out-of-contract attribute types) must fail loudly.

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
