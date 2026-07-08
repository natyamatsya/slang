//TEST:SIMPLE(filecheck=METAL): -target metal -metal-rt-max-payload-size 128

// -metal-rt-max-payload-size resizes the context's payload blob (part of
// the cross-stage ABI: every module linked into one pipeline must use the
// same value).
// METAL: array<uchar, int(128)> payload;

RWStructuredBuffer<float4> outputBuffer;

struct BigPayload
{
    float4 a;
    float4 b;
    float4 c;
    float4 d;
    float4 e; // 80 bytes: over the 64-byte default, within 128
};

[shader("raygeneration")]
void rayGenMain()
{
    outputBuffer[DispatchRaysIndex().x] = float4(0, 0, 0, 1);
}

[shader("miss")]
void missMain(inout BigPayload payload)
{
    payload.a = float4(1, 2, 3, 4);
}
