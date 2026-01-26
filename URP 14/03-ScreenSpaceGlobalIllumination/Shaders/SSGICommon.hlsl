#ifndef SSGI_COMMON_HLSL
#define SSGI_COMMON_HLSL

#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/UnityGBuffer.hlsl"

//=============================================================================
// Configuration
//=============================================================================
#define SSGI_TILE_SIZE 8
#define HZB_MIP_COUNT 10
#define MAX_ACCUM_FRAME_NUM 8
#define MAX_REPROJECTION_DISTANCE 0.1
#define MAX_PIXEL_TOLERANCE 4
#define PROJECTION_EPSILON 0.000001
#define RAW_FAR_CLIP_THRESHOLD 1e-7
#define CLAMP_MAX 65472.0

// GBuffer material flags
#define kMaterialFlagSpecularSetup 8

//=============================================================================
// Constant Buffers
//=============================================================================
CBUFFER_START(SSGIParams)
    float4 _SSGITextureSizes;           // xy: full res, zw: 1/full res
    float4 _SSGIHZBTextureSizes;        // xy: hzb res, zw: 1/hzb res
    float4 _SSGIIndirectTextureSizes;   // xy: indirect res, zw: 1/indirect res
    
    float _MaxSteps;
    float _MaxSmallSteps;
    float _MaxMediumSteps;
    float _StepSize;
    float _SmallStepSize;
    float _MediumStepSize;
    float _Thickness;
    float _ThicknessIncrement;
    
    float _RayCount;
    float _TemporalIntensity;
    float _MaxBrightness;
    float _DownSample;
    
    float _HistoryTextureValid;
    float _IsProbeCamera;
    float _BackDepthEnabled;
    float _FrameIndex;
    
    float _PixelSpreadAngleTangent;
    float _IndirectDiffuseLightingMultiplier;
    uint _IndirectDiffuseRenderingLayers;
    float _AggressiveDenoise;
    
    float4 _ReBlurBlurRotator;
    float _ReBlurDenoiserRadius;
    
    float4x4 _PrevInvViewProjMatrix;
    float3 _PrevCameraPositionWS;
    float _SSGIPadding0;
    
    // Ambient SH coefficients
    float4 _SSGI_SHAr;
    float4 _SSGI_SHAg;
    float4 _SSGI_SHAb;
    float4 _SSGI_SHBr;
    float4 _SSGI_SHBg;
    float4 _SSGI_SHBb;
    float4 _SSGI_SHC;
CBUFFER_END

//=============================================================================
// Textures - Input
//=============================================================================
Texture2D<float> _CameraDepthTexture;
Texture2D<float4> _CameraColorTexture;
Texture2D<float2> _MotionVectorTexture;
Texture2D<float4> _GBuffer0;
Texture2D<float4> _GBuffer1;
Texture2D<float4> _GBuffer2;
Texture2D<float> _CameraBackDepthTexture;
Texture2D<float4> _CameraBackOpaqueTexture;

// HZB Pyramid
Texture2D<float> _HZBTexture;

// History textures
Texture2D<float> _SSGIHistoryDepthTexture;
Texture2D<float4> _SSGIHistoryCameraColorTexture;
Texture2D<float4> _HistoryIndirectDiffuseTexture;
Texture2D<float> _SSGIHistorySampleTexture;

// Intermediate textures
Texture2D<float4> _SSGIInputTexture;
Texture2D<float4> _IndirectDiffuseTexture;
Texture2D<float> _SSGISampleTexture;
Texture2D<float4> _APVLightingTexture;

//=============================================================================
// Textures - Output (RWTexture2D)
//=============================================================================
RWTexture2D<float> _HZBOutput;
RWTexture2D<float4> _SSGIOutput;
RWTexture2D<float> _SSGISampleOutput;
RWTexture2D<float4> _DirectLightingOutput;
RWTexture2D<float4> _APVLightingOutput;
RWTexture2D<float> _HistoryDepthOutput;
RWTexture2D<float4> _HistoryColorOutput;
RWTexture2D<float4> _HistoryIndirectOutput;
RWTexture2D<float> _HistorySampleOutput;

//=============================================================================
// Reflection Probes (for ray miss fallback)
//=============================================================================
#if defined(_FP_REFL_PROBE_ATLAS)
    Texture2D<float4> _ReflProbeAtlas;
    SamplerState sampler_ReflProbeAtlas;
    
    StructuredBuffer<float4> _ReflProbeData; // BoxMin, BoxMax, ProbePosition, MipScaleOffset
    uint _ReflProbeCount;
#else
    TextureCube<float4> _SpecCube0;
    SamplerState sampler_SpecCube0;
    float4 _SpecCube0_HDR;
    float4 _SpecCube0_ProbePosition;
    float3 _SpecCube0_BoxMin;
    float3 _SpecCube0_BoxMax;
    float _ProbeWeight;
    float _ProbeSet;
#endif

//=============================================================================
// Utility Functions
//=============================================================================
float SSGIFastSign(float x)
{
    return x >= 0.0 ? 1.0 : -1.0;
}

float ConvertLinearEyeDepth(float deviceDepth)
{
    if (IsPerspectiveProjection())
        return LinearEyeDepth(deviceDepth, _ZBufferParams);
    else
    {
    #if UNITY_REVERSED_Z
        deviceDepth = 1.0 - deviceDepth;
    #endif
        return lerp(_ProjectionParams.y, _ProjectionParams.z, deviceDepth);
    }
}

float3 ComputeWorldSpacePositionFromDepth(float2 uv, float depth)
{
    return ComputeWorldSpacePosition(uv, depth, UNITY_MATRIX_I_VP);
}

bool IsBackground(float depth)
{
    return abs(depth - UNITY_RAW_FAR_CLIP_VALUE) < RAW_FAR_CLIP_THRESHOLD;
}

float3 DecodeNormalFromGBuffer(float3 normalData)
{
#if defined(_GBUFFER_NORMALS_OCT)
    float2 remapped = Unpack888ToFloat2(normalData);
    float2 octNormal = remapped * 2.0 - 1.0;
    return UnpackNormalOctQuadEncode(octNormal);
#else
    return normalize(normalData);
#endif
}

// Hash-based random number generation
float SSGIGenerateHashedRandomFloat(uint3 seed)
{
    uint hash = seed.x;
    hash = hash * 747796405u + 2891336453u;
    hash = ((hash >> ((hash >> 28u) + 4u)) ^ hash) * 277803737u;
    hash = (hash >> 22u) ^ hash;
    
    hash ^= seed.y;
    hash = hash * 747796405u + 2891336453u;
    hash = ((hash >> ((hash >> 28u) + 4u)) ^ hash) * 277803737u;
    hash = (hash >> 22u) ^ hash;
    
    hash ^= seed.z;
    hash = hash * 747796405u + 2891336453u;
    hash = ((hash >> ((hash >> 28u) + 4u)) ^ hash) * 277803737u;
    hash = (hash >> 22u) ^ hash;
    
    return float(hash) / 4294967295.0;
}

float GenerateRandomValue(float2 uv, inout float seed)
{
    seed += 1.0;
    return SSGIGenerateHashedRandomFloat(uint3(uv * _SSGITextureSizes.xy, _FrameIndex + seed));
}

// Cosine-weighted hemisphere sampling
float3 SSGISampleHemisphereCosine(float u1, float u2, float3 normal)
{
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    
    float3 tangentSpace = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
    
    // Build TBN matrix
    float3 up = abs(normal.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    
    return normalize(tangent * tangentSpace.x + bitangent * tangentSpace.y + normal * tangentSpace.z);
}

// Color space conversions
float3 SSGIRgbToHsv(float3 c)
{
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
    
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float3 SSGIHsvToRgb(float3 c)
{
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
}

// SH evaluation for ambient lighting
float3 EvaluateAmbientSH(float3 normal)
{
    float4 shAr = _SSGI_SHAr;
    float4 shAg = _SSGI_SHAg;
    float4 shAb = _SSGI_SHAb;
    float4 shBr = _SSGI_SHBr;
    float4 shBg = _SSGI_SHBg;
    float4 shBb = _SSGI_SHBb;
    float4 shC = _SSGI_SHC;
    
    // Linear + constant polynomial terms
    float3 res = SHEvalLinearL0L1(normal, shAr, shAg, shAb);
    
    // Quadratic polynomials
    res += SHEvalLinearL2(normal, shBr, shBg, shBb, shC);
    
#ifdef UNITY_COLORSPACE_GAMMA
    res = LinearToSRGB(res);
#endif
    
    return res;
}

// Metallic from reflectivity (for specular setup materials)
float MetallicFromReflectivity(float reflectivity)
{
    float oneMinusDielectricSpec = 1.0 - 0.04;
    return (reflectivity - 0.04) / oneMinusDielectricSpec;
}

float ReflectivitySpecular(float3 specular)
{
    return max(max(specular.r, specular.g), specular.b);
}

// Clamp to prevent fireflies
float3 ClampBrightness(float3 color, float maxBrightness)
{
    float3 hsv = SSGIRgbToHsv(color);
    hsv.z = clamp(hsv.z, 0.0, maxBrightness);
    return SSGIHsvToRgb(hsv);
}

//=============================================================================
// Structures
//=============================================================================
struct Ray
{
    float3 origin;
    float3 direction;
};

struct RayHit
{
    float3 position;
    float distance;
    float3 normal;
    float3 emission;
    bool isValid;
};

RayHit CreateRayHit()
{
    RayHit hit;
    hit.position = 0;
    hit.distance = 0;
    hit.normal = 0;
    hit.emission = 0;
    hit.isValid = false;
    return hit;
}

//=============================================================================
// Reprojection Helper
//=============================================================================
float ComputeMaxReprojectionWorldRadius(float3 positionWS, float3 viewDir, float3 normal)
{
    float parallelPixelFootPrint = _PixelSpreadAngleTangent * length(positionWS - GetCameraPositionWS());
    float realPixelFootPrint = parallelPixelFootPrint / max(abs(dot(normal, viewDir)), PROJECTION_EPSILON);
    return max(MAX_REPROJECTION_DISTANCE, realPixelFootPrint * MAX_PIXEL_TOLERANCE);
}

#endif // SSGI_COMMON_HLSL
