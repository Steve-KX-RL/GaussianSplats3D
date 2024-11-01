#version 300 es
#define varying in
layout(location = 0) out highp vec4 pc_fragColor;
#define gl_FragColor pc_fragColor
#define gl_FragDepthEXT gl_FragDepth
#define texture2D texture
#define textureCube texture
#define texture2DProj textureProj
#define texture2DLodEXT textureLod
#define texture2DProjLodEXT textureProjLod
#define textureCubeLodEXT textureLod
#define texture2DGradEXT textureGrad
#define texture2DProjGradEXT textureProjGrad
#define textureCubeGradEXT textureGrad
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
#define USE_ALPHATEST
#define DOUBLE_SIDED
uniform mat4 viewMatrix;
uniform vec3 cameraPosition;
uniform bool isOrthographic;
vec4 LinearTransferOETF(in vec4 value) {
    return value;
}
vec4 sRGBTransferEOTF(in vec4 value) {
    return vec4(mix(pow(value.rgb * 0.9478672986f + vec3(0.0521327014f), vec3(2.4f)), value.rgb * 0.0773993808f, vec3(lessThanEqual(value.rgb, vec3(0.04045f)))), value.a);
}
vec4 sRGBTransferOETF(in vec4 value) {
    return vec4(mix(pow(value.rgb, vec3(0.41666f)) * 1.055f - vec3(0.055f), value.rgb * 12.92f, vec3(lessThanEqual(value.rgb, vec3(0.0031308f)))), value.a);
}
vec4 linearToOutputTexel(vec4 value) {
    return sRGBTransferOETF(vec4(value.rgb * mat3(1.0000f, -0.0000f, -0.0000f, -0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f), value.a));
}
float luminance(const in vec3 rgb) {
    const vec3 weights = vec3(0.2126f, 0.7152f, 0.0722f);
    return dot(weights, rgb);
}
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

uniform vec3 debugColor;
varying vec4 vColor;
varying vec2 vUv;
varying vec2 vPosition;
void main() {
// Compute the positional squared distance from the center of the splat to the current fragment.
    float A = dot(vPosition, vPosition);
// Since the positional data in vPosition has been scaled by sqrt(8), the squared result will be

// scaled by a factor of 8. If the squared result is larger than 8, it means it is outside the ellipse
// defined by the rectangle formed by vPosition. It also means it's farther
// away than sqrt(8) standard deviations from the mean.
    if(A > 8.0f)
        discard;
    vec3 color = vColor.rgb;

// Since the rendered splat is scaled by sqrt(8), the inverse covariance matrix that is part of

// the gaussian formula becomes the identity matrix. We're then left with (X - mean) * (X - mean), // and since 'mean' is zero, we have X * X, which is the same as A:
    float opacity = exp(-0.5f * A) * vColor.a;
    gl_FragColor = vec4(color.rgb, opacity);
}
