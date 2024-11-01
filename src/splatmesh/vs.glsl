#version 300 es

#define attribute in
#define varying out
#define texture2D texture
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp samplerCube;
precision highp sampler3D;
precision highp sampler2DArray;
precision highp sampler2DShadow;
precision highp samplerCubeShadow;
precision highp sampler2DArrayShadow;
precision highp isampler2D;
precision highp isampler3D;
precision highp isamplerCube;
precision highp isampler2DArray;
precision highp usampler2D;
precision highp usampler3D;
precision highp usamplerCube;
precision highp usampler2DArray;
#define HIGH_PRECISION
#define SHADER_TYPE ShaderMaterial
#define SHADER_NAME
#define DOUBLE_SIDED
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;
uniform bool isOrthographic;
#ifdef USE_INSTANCING
attribute mat4 instanceMatrix;
#endif
#ifdef USE_INSTANCING_COLOR
attribute vec3 instanceColor;
#endif
#ifdef USE_INSTANCING_MORPH
uniform sampler2D morphTexture;
#endif
attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;
#ifdef USE_UV1
attribute vec2 uv1;
#endif
#ifdef USE_UV2
attribute vec2 uv2;
#endif
#ifdef USE_UV3
attribute vec2 uv3;
#endif
#ifdef USE_TANGENT
attribute vec4 tangent;
#endif
#if defined( USE_COLOR_ALPHA )
attribute vec4 color;
#elif defined( USE_COLOR )
attribute vec3 color;
#endif
#ifdef USE_SKINNING
attribute vec4 skinIndex;
attribute vec4 skinWeight;
#endif

precision highp float;
#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
    #define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2(const in float x) {
    return x * x;
}
vec3 pow2(const in vec3 x) {
    return x * x;
}
float pow3(const in float x) {
    return x * x * x;
}
float pow4(const in float x) {
    float x2 = x * x;
    return x2 * x2;
}
float max3(const in vec3 v) {
    return max(max(v.x, v.y), v.z);
}
float average(const in vec3 v) {
    return dot(v, vec3(0.3333333f));
}
highp float rand(const in vec2 uv) {
    const highp float a = 12.9898f, b = 78.233f, c = 43758.5453f;
    highp float dt = dot(uv.xy, vec2(a, b)), sn = mod(dt, PI);
    return fract(sin(sn) * c);
}
#ifdef HIGH_PRECISION
float precisionSafeLength(vec3 v) {
    return length(v);
}
#else
float precisionSafeLength(vec3 v) {
    float maxComponent = max3(abs(v));
    return length(v / maxComponent) * maxComponent;
}
#endif
struct IncidentLight {
    vec3 color;
    vec3 direction;
    bool visible;
};
struct ReflectedLight {
    vec3 directDiffuse;
    vec3 directSpecular;
    vec3 indirectDiffuse;
    vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
varying vec3 vPosition;
#endif
vec3 transformDirection(in vec3 dir, in mat4 matrix) {
    return normalize((matrix * vec4(dir, 0.0f)).xyz);
}
vec3 inverseTransformDirection(in vec3 dir, in mat4 matrix) {
    return normalize((vec4(dir, 0.0f) * matrix).xyz);
}
mat3 transposeMat3(const in mat3 m) {
    mat3 tmp;
    tmp[0] = vec3(m[0].x, m[1].x, m[2].x);
    tmp[1] = vec3(m[0].y, m[1].y, m[2].y);
    tmp[2] = vec3(m[0].z, m[1].z, m[2].z);
    return tmp;
}
bool isPerspectiveMatrix(mat4 m) {
    return m[2][3] == -1.0f;
}
vec2 equirectUv(in vec3 dir) {
    float u = atan(dir.z, dir.x) * RECIPROCAL_PI2 + 0.5f;
    float v = asin(clamp(dir.y, -1.0f, 1.0f)) * RECIPROCAL_PI + 0.5f;
    return vec2(u, v);
}
vec3 BRDF_Lambert(const in vec3 diffuseColor) {
    return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick(const in vec3 f0, const in float f90, const in float dotVH) {
    float fresnel = exp2((-5.55473f * dotVH - 6.98316f) * dotVH);
    return f0 * (1.0f - fresnel) + (f90 * fresnel);
}
float F_Schlick(const in float f0, const in float f90, const in float dotVH) {
    float fresnel = exp2((-5.55473f * dotVH - 6.98316f) * dotVH);
    return f0 * (1.0f - fresnel) + (f90 * fresnel);
}
// validated

attribute uint splatIndex;
uniform highp usampler2D centersColorsTexture;
uniform highp sampler2D sphericalHarmonicsTexture;
uniform highp sampler2D sphericalHarmonicsTextureR;
uniform highp sampler2D sphericalHarmonicsTextureG;
uniform highp sampler2D sphericalHarmonicsTextureB;
uniform highp usampler2D sceneIndexesTexture;
uniform vec2 sceneIndexesTextureSize;
uniform int sceneCount;
uniform vec2 covariancesTextureSize;
uniform highp sampler2D covariancesTexture;
uniform highp usampler2D covariancesTextureHalfFloat;
uniform int covariancesAreHalfFloat;
void fromCovarianceHalfFloatV4(uvec4 val, out vec4 first, out vec4 second) {
    vec2 r = unpackHalf2x16(val.r);
    vec2 g = unpackHalf2x16(val.g);
    vec2 b = unpackHalf2x16(val.b);
    first = vec4(r.x, r.y, g.x, g.y);
    second = vec4(b.x, b.y, 0.0f, 0.0f);
}
uniform vec2 focal;
uniform float orthoZoom;
uniform int orthographicMode;
uniform int pointCloudModeEnabled;
uniform float inverseFocalAdjustment;
uniform vec2 viewport;
uniform vec2 basisViewport;
uniform vec2 centersColorsTextureSize;
uniform int sphericalHarmonicsDegree;
uniform vec2 sphericalHarmonicsTextureSize;
uniform int sphericalHarmonics8BitMode;
uniform int sphericalHarmonicsMultiTextureMode;
uniform float visibleRegionRadius;
uniform float visibleRegionFadeStartRadius;
uniform float firstRenderTime;
uniform float currentTime;
uniform int fadeInComplete;
uniform vec3 sceneCenter;
uniform float splatScale;
uniform float sphericalHarmonics8BitCompressionRangeMin[32];
uniform float sphericalHarmonics8BitCompressionRangeMax[32];
varying vec4 vColor;
varying vec2 vUv;
varying vec2 vPosition;
mat3 quaternionToRotationMatrix(float x, float y, float z, float w) {
    float s = 1.0f / sqrt(w * w + x * x + y * y + z * z);
    return mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y), 2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x), 2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + y * y));
}
const float sqrt8 = sqrt(8.0f);
const float minAlpha = 1.0f / 255.0f;
const vec4 encodeNorm4 = vec4(1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f);
const uvec4 mask4 = uvec4(uint(0x000000FF), uint(0x0000FF00), uint(0x00FF0000), uint(0xFF000000));
const uvec4 shift4 = uvec4(0, 8, 16, 24);
vec4 uintToRGBAVec(uint u) {
    uvec4 urgba = mask4 & u;
    urgba = urgba >> shift4;
    vec4 rgba = vec4(urgba) * encodeNorm4;
    return rgba;
}
vec2 getDataUV(in int stride, in int offset, in vec2 dimensions) {
    vec2 samplerUV = vec2(0.0f, 0.0f);
    float d = float(splatIndex * uint(stride) + uint(offset)) / dimensions.x;
    samplerUV.y = float(floor(d)) / dimensions.y;
    samplerUV.x = fract(d);
    return samplerUV;
}
vec2 getDataUVF(in uint sIndex, in float stride, in uint offset, in vec2 dimensions) {
    vec2 samplerUV = vec2(0.0f, 0.0f);
    float d = float(uint(float(sIndex) * stride) + offset) / dimensions.x;
    samplerUV.y = float(floor(d)) / dimensions.y;
    samplerUV.x = fract(d);
    return samplerUV;
}
const float SH_C1 = 0.4886025119029199f;
const float[5] SH_C2 = float[](1.0925484f, -1.0925484f, 0.3153916f, -1.0925484f, 0.5462742f);
void main() {
    uint oddOffset = splatIndex & uint(0x00000001);
    uint doubleOddOffset = oddOffset * uint(2);
    bool isEven = oddOffset == uint(0);
    uint nearestEvenIndex = splatIndex - oddOffset;
    float fOddOffset = float(oddOffset);
    uvec4 sampledCenterColor = texture(centersColorsTexture, getDataUV(1, 0, centersColorsTextureSize));
    vec3 splatCenter = uintBitsToFloat(uvec3(sampledCenterColor.gba));
    uint sceneIndex = uint(0);
    if(sceneCount > 1) {
        sceneIndex = texture(sceneIndexesTexture, getDataUV(1, 0, sceneIndexesTextureSize)).r;
    }
    mat4 transformModelViewMatrix = modelViewMatrix;
    float sh8BitCompressionRangeMinForScene = sphericalHarmonics8BitCompressionRangeMin[sceneIndex];
    float sh8BitCompressionRangeMaxForScene = sphericalHarmonics8BitCompressionRangeMax[sceneIndex];
    float sh8BitCompressionRangeForScene = sh8BitCompressionRangeMaxForScene - sh8BitCompressionRangeMinForScene;
    float sh8BitCompressionHalfRangeForScene = sh8BitCompressionRangeForScene / 2.0f;
    vec3 vec8BitSHShift = vec3(sh8BitCompressionRangeMinForScene);
    vec4 viewCenter = transformModelViewMatrix * vec4(splatCenter, 1.0f);
    vec4 clipCenter = projectionMatrix * viewCenter;
    float clip = 1.2f * clipCenter.w;
    if(clipCenter.z < -clip || clipCenter.x < -clip || clipCenter.x > clip || clipCenter.y < -clip || clipCenter.y > clip) {
        gl_Position = vec4(0.0f, 0.0f, 2.0f, 1.0f);
        return;
    }
    vec3 ndcCenter = clipCenter.xyz / clipCenter.w;
    vPosition = position.xy;
    vColor = uintToRGBAVec(sampledCenterColor.r);
    vec4 sampledCovarianceA;
    vec4 sampledCovarianceB;
    vec3 cov3D_M11_M12_M13;
    vec3 cov3D_M22_M23_M33;
    if(covariancesAreHalfFloat == 0) {
        sampledCovarianceA = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5f, oddOffset, covariancesTextureSize));
        sampledCovarianceB = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5f, oddOffset + uint(1), covariancesTextureSize));
        cov3D_M11_M12_M13 = vec3(sampledCovarianceA.rgb) * (1.0f - fOddOffset) +
            vec3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;
        cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0f - fOddOffset) +
            vec3(sampledCovarianceB.gba) * fOddOffset;
    } else {
        uvec4 sampledCovarianceU = texture(covariancesTextureHalfFloat, getDataUV(1, 0, covariancesTextureSize));
        fromCovarianceHalfFloatV4(sampledCovarianceU, sampledCovarianceA, sampledCovarianceB);
        cov3D_M11_M12_M13 = sampledCovarianceA.rgb;
        cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg);
    }
// Construct the 3D covariance matrix
    mat3 Vrk = mat3(cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z, cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x, cov3D_M22_M23_M33.y, cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z);
    mat3 J;
    if(orthographicMode == 1) {
// Since the projection is linear, we don't need an approximation
        J = transpose(mat3(orthoZoom, 0.0f, 0.0f, 0.0f, orthoZoom, 0.0f, 0.0f, 0.0f, 0.0f));
    } else {
// Construct the Jacobian of the affine approximation of the projection matrix. It will be used to transform the
// 3D covariance matrix instead of using the actual projection matrix because that transformation would
// require a non-linear component (perspective division) which would yield a non-gaussian result.
        float s = 1.0f / (viewCenter.z * viewCenter.z);
        J = mat3(focal.x / viewCenter.z, 0.f, -(focal.x * viewCenter.x) * s, 0.f, focal.y / viewCenter.z, -(focal.y * viewCenter.y) * s, 0.f, 0.f, 0.f);
    }
// Concatenate the projection approximation with the model-view transformation
    mat3 W = transpose(mat3(transformModelViewMatrix));
    mat3 T = W * J;

// Transform the 3D covariance matrix (Vrk) to compute the 2D covariance matrix

    mat3 cov2Dm = transpose(T) * Vrk * T;
    cov2Dm[0][0] += 0.3f;
    cov2Dm[1][1] += 0.3f;

// We are interested in the upper-left 2x2 portion of the projected 3D covariance matrix because

// we only care about the X and Y values. We want the X-diagonal, cov2Dm[0][0], // the Y-diagonal, cov2Dm[1][1], and the correlation between the two cov2Dm[0][1]. We don't
// need cov2Dm[1][0] because it is a symetric matrix.
    vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

// We now need to solve for the eigen-values and eigen vectors of the 2D covariance matrix

// so that we can determine the 2D basis for the splat. This is done using the method described
// here: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
// After calculating the eigen-values and eigen-vectors, we calculate the basis for rendering the splat
// by normalizing the eigen-vectors and then multiplying them by (sqrt(8) * sqrt(eigen-value)), which is
// equal to scaling them by sqrt(8) standard deviations.
//
// This is a different approach than in the original work at INRIA. In that work they compute the
// max extents of the projected splat in screen space to form a screen-space aligned bounding rectangle
// which forms the geometry that is actually rasterized. The dimensions of that bounding box are 3.0
// times the square root of the maximum eigen-value, or 3 standard deviations. They then use the inverse
// 2D covariance matrix (called 'conic') in the CUDA rendering thread to determine fragment opacity by
// calculating the full gaussian: exp(-0.5 * (X - mean) * conic * (X - mean)) * splat opacity
    float a = cov2Dv.x;
    float d = cov2Dv.z;
    float b = cov2Dv.y;
    float D = a * d - b * b;
    float trace = a + d;
    float traceOver2 = 0.5f * trace;
    float term2 = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
    float eigenValue1 = traceOver2 + term2;
    float eigenValue2 = traceOver2 - term2;
    if(pointCloudModeEnabled == 1) {
        eigenValue1 = eigenValue2 = 0.2f;
    }
    if(eigenValue2 <= 0.0f)
        return;
    vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
// since the eigen vectors are orthogonal, we derive the second one from the first

    vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

// We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.

    vec2 basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), 1024.0f);
    vec2 basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), 1024.0f);
    vec2 ndcOffset = vec2(vPosition.x * basisVector1 + vPosition.y * basisVector2) *
        basisViewport * 2.0f * inverseFocalAdjustment;
    vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0f);
    gl_Position = quadPos;

// Scale the position data we send to the fragment shader

    vPosition *= sqrt8;
    if(fadeInComplete == 0) {
        float opacityAdjust = 1.0f;
        float centerDist = length(splatCenter - sceneCenter);
        float renderTime = max(currentTime - firstRenderTime, 0.0f);
        float fadeDistance = 0.75f;
        float distanceLoadFadeInFactor = step(visibleRegionFadeStartRadius, centerDist);
        distanceLoadFadeInFactor = (1.0f - distanceLoadFadeInFactor) +
            (1.0f - clamp((centerDist - visibleRegionFadeStartRadius) / fadeDistance, 0.0f, 1.0f)) *
            distanceLoadFadeInFactor;
        opacityAdjust *= distanceLoadFadeInFactor;
        vColor.a *= opacityAdjust;
    }

}
