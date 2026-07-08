//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target metal -metal-rt-globals-slots 8 -entry missMain -stage miss

// The slot-addressed globals layout (runtime-contract spec section 3)
// diagnoses every way a resource can fail to own a valid slot: no explicit
// register (the register is the slot address), a register outside the slot
// range, two resources addressing one slot ('t' and 'u' share the index
// space), and register spaces (unsupported in v1).

StructuredBuffer<float4> noReg;
StructuredBuffer<float4> tooBig : register(t9);
StructuredBuffer<float4> dupA : register(t3);
RWStructuredBuffer<float4> dupB : register(u3);
StructuredBuffer<float4> spaced : register(t1, space2);
SamplerState samp : register(s0);
Texture2D<float4> tex : register(t2);

struct RayPayload { float4 color; };

//CHECK-DAG: E56118
//CHECK-DAG: 'noReg' is used by a ray-tracing handler stage but has no explicit 'register()'
//CHECK-DAG: E56119
//CHECK-DAG: 'tooBig' is bound at register 9, which is outside the 8 slots
//CHECK-DAG: E56120
//CHECK-DAG: 'dupB' and 'dupA' both address slot 3
//CHECK-DAG: E56121
//CHECK-DAG: 'spaced' uses a register space
//CHECK-DAG: E56122
//CHECK-DAG: 'samp' cannot be slot-addressed
[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.color = noReg[0] + tooBig[0] + dupA[0] + dupB[0] + spaced[0] +
                    tex.SampleLevel(samp, float2(0.5, 0.5), 0);
}
