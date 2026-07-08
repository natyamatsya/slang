//TEST:SIMPLE(filecheck=METAL): -target metal

// Test the P2 Metal lowering of callable shaders
// (docs/design/metal-raytracing.md sections 5.2 and 10): CallShader
// dispatches the record `callableBase + shaderIndex` through the visible
// function table with the callable data bridged through the context's
// payload blob, and a callable shader becomes a [[visible]] function with
// the same uniform handler signature as miss/closest-hit shaders.

RWStructuredBuffer<float4> outputBuffer;

struct ShadeData
{
    float3 albedo;
    float brightness;
};

// CallShader packs the callable data into the payload blob, dispatches
// callableBase + 2, and unpacks the blob back:
// METAL: {{\[\[}}kernel{{\]\]}} void rayGenMain(
// METAL: (ShadeData{{[^)]*}} thread*)(&(&slang_rtCtx{{[^)]*}})->payload)
// METAL: slang_rtHandlers{{[^[]*}}[{{[^]]*}}->callableBase{{[^]]*}}](&slang_rtCtx{{[^,]*}}, slang_rtGlobals{{[^)]*}})
[shader("raygeneration")]
void rayGenMain()
{
    uint3 idx = DispatchRaysIndex();
    ShadeData shade;
    shade.albedo = float3(idx.x, idx.y, 0.5);
    shade.brightness = 0.0;
    CallShader(2, shade);
    outputBuffer[idx.x] = float4(shade.albedo * shade.brightness, 1.0);
}

// The callable shader has the uniform handler signature and bridges its
// data through the payload blob like a miss shader does:
// METAL: {{\[\[}}visible{{\]\]}} void tonemapMain(slang_RTContext{{[^,]*}} thread* slang_rtCtx{{[^,]*}}, slang_RTGlobals{{[^,]*}} device* slang_rtGlobals{{[^)]*}})
// METAL: (ShadeData{{[^)]*}} thread*)(&slang_rtCtx{{[^)]*}}->payload)
[shader("callable")]
void tonemapMain(inout ShadeData data)
{
    data.brightness = dot(data.albedo, float3(0.299, 0.587, 0.114)) + float(DispatchRaysIndex().x);
    data.albedo = data.albedo / (1.0 + data.albedo);
}
