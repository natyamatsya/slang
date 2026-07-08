//TEST:SIMPLE(filecheck=LEAN): -target metal
//TEST:SIMPLE(filecheck=OPTED): -target metal -metal-rt-force-world-space-data -metal-rt-force-isect-table -metal-rt-max-payload-size 128
//TEST:SIMPLE(filecheck=HANDLER): -target metal -entry missMain -stage miss

// The machine-readable ABI descriptor of the runtime contract
// (docs/design/metal-raytracing-runtime-contract.md section 2): one strict
// JSON line per RT-stage module; scalar fields always present in fixed
// order, `sys` in ray-generation modules, `uniforms` when the module
// references loose uniforms.
//
// LEAN: // slang-metal-rt-abi:{"v":1,"payload":64,"attr":32,"ws":0,"isect":0,"slots":0,"sys":{"handlers":30,"sbt":29,"globals":28,"isect":27},"uniforms":{"buf":0,"size":16}}
// OPTED: // slang-metal-rt-abi:{"v":1,"payload":128,"attr":32,"ws":1,"isect":1,"slots":0,"sys":{"handlers":30,"sbt":29,"globals":28,"isect":27},"uniforms":{"buf":0,"size":16}}
//
// A handler-only module — the separate-compilation shape the contract
// exists for — has no `sys` (no kernel) and a `buf`-less `uniforms`:
// HANDLER: // slang-metal-rt-abi:{"v":1,"payload":64,"attr":32,"ws":0,"isect":0,"slots":0,"uniforms":{"size":16}}

uniform float4 u_tint;
uniform RaytracingAccelerationStructure scene;
RWStructuredBuffer<float4> outputBuffer;

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
    outputBuffer[idx.x] = payload.color * u_tint;
}

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.color = float4(0.1, 0.2, 0.3, 1.0) * u_tint;
}
