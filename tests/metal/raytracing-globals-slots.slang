//TEST:SIMPLE(filecheck=SLOTS): -target metal -metal-rt-globals-slots 8 -warnings-disable 39029
//TEST:SIMPLE(filecheck=HANDLER): -target metal -metal-rt-globals-slots 8 -entry missMain -stage miss -warnings-disable 39029

// Test the slot-addressed globals layout (runtime-contract spec section 3):
// under -metal-rt-globals-slots N the tail of slang_RTGlobals is exactly N
// 8-byte slots addressed by declared register() number — typed where this
// module's handlers reference the resource, opaque placeholders elsewhere —
// and the fixed header ends with the implicit-constant-buffer `uniforms`
// entry. A handler-only compile (the separate-compilation case) produces
// the identical layout by construction.
//
// SLOTS: // slang-metal-rt-abi:{{.*}}"slots":8,"sys":{{.*}}"uniforms":{"buf":1,"size":32}}
// SLOTS: struct slang_RTGlobals
// SLOTS: slang_RTSbt sbt;
// SLOTS-NEXT: GlobalParams{{[^ ]*}} constant* uniforms;
// SLOTS-NEXT: ulong slang_rtSlot0;
// SLOTS-NEXT: ulong slang_rtSlot1;
// SLOTS-NEXT: ulong slang_rtSlot2;
// SLOTS-NEXT: ulong slang_rtSlot3;
// SLOTS-NEXT: ulong slang_rtSlot4;
// SLOTS-NEXT: packed_float4 device* bufA;
// SLOTS-NEXT: ulong slang_rtSlot6;
// SLOTS-NEXT: acceleration_structure{{.*}} scene;
// SLOTS-NEXT: };
//
// HANDLER: // slang-metal-rt-abi:{{.*}}"slots":8,"uniforms":{"size":32}}
// HANDLER: slang_RTSbt sbt;
// HANDLER-NEXT: GlobalParams{{[^ ]*}} constant* uniforms;
// HANDLER-NEXT: ulong slang_rtSlot0;
// HANDLER: packed_float4 device* bufA;
// HANDLER: ulong slang_rtSlot7;

uniform float4 u_params[2];
StructuredBuffer<float4> bufA : register(t5);
RWStructuredBuffer<float4> outputBuffer : register(u0);
uniform RaytracingAccelerationStructure scene : register(t7);

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

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.color = bufA[0] * u_params[1];
}

// The closest-hit shader traces again, so `scene` is handler-referenced and
// its slot (7) is typed in both compiles that include this entry point.
[shader("closesthit")]
void chitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attrs)
{
    RayDesc ray;
    ray.Origin = float3(0, 0, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.01;
    ray.TMax = 10.0;
    RayPayload inner;
    inner.color = float4(1, 1, 1, 1);
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, inner);
    payload.color = inner.color + bufA[1];
}
