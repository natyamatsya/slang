// slang-emit-metal-prelude.cpp
#include "slang-emit-metal.h"

namespace Slang
{

const char* MetalSourceEmitter::kMetalBuiltinPreludeMatrixCompMult = R"(
template<typename T, int A, int B>
matrix<T,A,B> _slang_matrixCompMult(matrix<T,A,B> m1, matrix<T,A,B> m2)
{
    matrix<T,A,B> result;
    for (int i = 0; i < A; i++)
        result[i] = m1[i] * m2[i];
    return result;
}
)";

const char* MetalSourceEmitter::kMetalBuiltinPreludeMatrixReshape = R"(
template<int A, int B, typename T, int N, int M>
matrix<T,A,B> _slang_matrixReshape(matrix<T,N,M> m)
{
    matrix<T,A,B> result = T(0);
    for (int i = 0; i < min(A,N); i++)
        for (int j = 0; j < min(B,M); j++)
            result[i] = m[i][j];
    return result;
}
)";

const char* MetalSourceEmitter::kMetalBuiltinPreludeVectorReshape = R"(
template<int A, typename T, int N>
vec<T,A> _slang_vectorReshape(vec<T,N> v)
{
    vec<T,A> result = T(0);
    for (int i = 0; i < min(A,N); i++)
        result[i] = v[i];
    return result;
}
)";

const char* MetalSourceEmitter::kMetalBuiltinPreludeMatrixFmod = R"(
template<typename T, int A, int B>
matrix<T,A,B> _slang_matrixFmod(matrix<T,A,B> m1, matrix<T,A,B> m2)
{
    matrix<T,A,B> result;
    for (int i = 0; i < A; i++)
        result[i] = fmod(m1[i], m2[i]);
    return result;
}
)";

const char* MetalSourceEmitter::kMetalBuiltinPreludeSimdgroupMatrixOps = R"(
#include <metal_simdgroup_matrix>
template<typename T, int Cols, int Rows, typename V>
void _slang_simdgroup_fill(thread simdgroup_matrix<T, Cols, Rows>* dest, V val) {
    *dest = make_filled_simdgroup_matrix<T, Cols, Rows>(T(val));
}
template<typename Matrix, typename T>
Matrix _slang_simdgroup_load(const device T* src, ulong elements_per_row) {
    Matrix result;
    simdgroup_load(result, src, elements_per_row);
    return result;
}
template<typename Matrix, typename T>
Matrix _slang_simdgroup_load_transpose(const device T* src, ulong elements_per_row) {
    Matrix result;
    simdgroup_load(result, src, elements_per_row, ulong2(0), true);
    return result;
}
template<typename Matrix, typename T>
Matrix _slang_simdgroup_load(const threadgroup T* src, ulong elements_per_row) {
    Matrix result;
    simdgroup_load(result, src, elements_per_row);
    return result;
}
template<typename Matrix, typename T>
Matrix _slang_simdgroup_load_transpose(const threadgroup T* src, ulong elements_per_row) {
    Matrix result;
    simdgroup_load(result, src, elements_per_row, ulong2(0), true);
    return result;
}
)";

// Included on demand rather than in the front matter, since it only exists from MSL 3.2 onwards.
const char* MetalSourceEmitter::kMetalBuiltinPreludeLogging = R"(
#include <metal_logging>
)";

// The traversal half of one TraceRay (docs/design/metal-raytracing.md
// section 5.1): configure an intersector from the DXR ray flags, run
// intersect() — through the intersection function table when the program
// links anyhit/intersection stages — and fill the committed-hit fields of
// the slang_RTContext. Returns the committed intersection type (0 = none,
// 1 = triangle, 2 = bounding box). Templated over the context type because
// the prelude is emitted before type declarations; the flag values are
// DXR's RAY_FLAG_* encoding. For bounding-box hits the hit kind, the
// attribute blob, and (per DXR's definition of RayTCurrent in a
// procedural-hit group) the barycentrics field are already maintained by
// the accepted ReportHit, so only the shared committed fields are written.
// The uniform handler signature of the ray-tracing ABI names the pinned
// context/globals structs from positions that don't carry IR dependency
// edges (e.g. inside `slang_RTGlobals` itself), so the emitted declarations
// can precede the struct definitions; C++ only needs the names declared.
const char* MetalSourceEmitter::kMetalBuiltinPreludeRTForwardDecls = R"(
struct slang_RTContext;
struct slang_RTGlobals;
)";

const char* MetalSourceEmitter::kMetalBuiltinPreludeRTTrace = R"(
inline void _slang_rtTraceConfigure(
    thread raytracing::intersector<raytracing::triangle_data, raytracing::instancing>& i,
    uint flags)
{
    if (flags & 0x01) /* RAY_FLAG_FORCE_OPAQUE */
        i.force_opacity(raytracing::forced_opacity::opaque);
    if (flags & 0x02) /* RAY_FLAG_FORCE_NON_OPAQUE */
        i.force_opacity(raytracing::forced_opacity::non_opaque);
    if (flags & 0x04) /* RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH */
        i.accept_any_intersection(true);
    if (flags & 0x40) /* RAY_FLAG_CULL_OPAQUE */
        i.set_opacity_cull_mode(raytracing::opacity_cull_mode::opaque);
    if (flags & 0x80) /* RAY_FLAG_CULL_NON_OPAQUE */
        i.set_opacity_cull_mode(raytracing::opacity_cull_mode::non_opaque);
    if (flags & 0x100) /* RAY_FLAG_SKIP_TRIANGLES */
        i.set_geometry_cull_mode(raytracing::geometry_cull_mode::triangle);
    if (flags & 0x200) /* RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES */
        i.set_geometry_cull_mode(raytracing::geometry_cull_mode::bounding_box);
}
template<typename Context, typename Result>
uint _slang_rtTraceCommit(thread Context* ctx, Result hit)
{
    if (hit.type == raytracing::intersection_type::none)
        return 0u;
    ctx->hitT = hit.distance;
    ctx->instanceIndex = hit.instance_id;
    ctx->instanceID = hit.user_instance_id;
    ctx->geometryIndex = hit.geometry_id;
    ctx->primitiveIndex = hit.primitive_id;
    if (hit.type == raytracing::intersection_type::triangle)
    {
        ctx->triBarycentrics = hit.triangle_barycentric_coord;
        ctx->hitKind = hit.triangle_front_facing ? 254u : 255u;
        return 1u;
    }
    return 2u;
}
template<typename Context>
uint _slang_rtTrace(
    thread Context* ctx,
    metal::raytracing::acceleration_structure<metal::raytracing::instancing> scene,
    raytracing::intersection_function_table<raytracing::triangle_data, raytracing::instancing> table,
    uint mask,
    uint flags)
{
    raytracing::ray r(ctx->origin, ctx->direction, ctx->tMin, ctx->tMax);
    raytracing::intersector<raytracing::triangle_data, raytracing::instancing> i;
    _slang_rtTraceConfigure(i, flags);
    return _slang_rtTraceCommit(ctx, i.intersect(r, scene, mask, table, *ctx));
}
template<typename Context>
uint _slang_rtTrace(
    thread Context* ctx,
    metal::raytracing::acceleration_structure<metal::raytracing::instancing> scene,
    uint mask,
    uint flags)
{
    raytracing::ray r(ctx->origin, ctx->direction, ctx->tMin, ctx->tMax);
    raytracing::intersector<raytracing::triangle_data, raytracing::instancing> i;
    _slang_rtTraceConfigure(i, flags);
    return _slang_rtTraceCommit(ctx, i.intersect(r, scene, mask));
}
)";

} // namespace Slang
