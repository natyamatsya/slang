//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target metal

// The Metal ray-tracing ABI carries the payload in a fixed 64-byte blob
// inside slang_RTContext (docs/design/metal-raytracing.md section 4.1); a
// payload type that cannot fit must be rejected at compile time.

struct HugePayload
{
    float4 a;
    float4 b;
    float4 c;
    float4 d;
    float4 e; // 80 bytes > 64
};

//CHECK: 56112
[shader("miss")]
void missMain(inout HugePayload payload)
{
    payload.a = float4(1, 0, 0, 1);
}
