//TEST:SIMPLE(filecheck=HLSL): -target hlsl -entry computeMain -stage compute

// A `select` on a composite (non-basic, non-vector, non-matrix) type is
// legalized into an if/else whose tail — every instruction after the select
// in the original block — must move into the merge block.
// `legalizeCompositeSelect` moved that tail with the receiver and argument
// of `IRInst::insertAtEnd` reversed, inserting the merge BLOCK into each
// tail INSTRUCTION and crashing the compiler on this shader.
//
// HLSL: void computeMain(
// HLSL: if(
// HLSL: else

struct Pair
{
    float a;
    float b;
};

RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID)
{
    Pair p1;
    p1.a = 1.0;
    p1.b = 2.0;
    Pair p2;
    p2.a = 3.0;
    p2.b = 4.0;
    Pair chosen = select(tid.x > 0, p1, p2);
    // The tail that must survive the move:
    float tail = chosen.a * 10.0 + chosen.b;
    outputBuffer[tid.x] = tail;
}
