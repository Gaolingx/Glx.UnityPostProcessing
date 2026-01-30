// SSGIComputeUtilities.hlsl
#ifndef SSGI_COMPUTE_UTILITIES_HLSL
#define SSGI_COMPUTE_UTILITIES_HLSL

#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/BRDF.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/SpaceTransforms.hlsl"

// ============================================================================
// Configuration
// ============================================================================

#define MAX_ACCUM_FRAME_NUM         8
#define MAX_REPROJECTION_DISTANCE   0.1
#define MAX_PIXEL_TOLERANCE         4
#define PROJECTION_EPSILON          0.000001
#define CLAMP_MAX                   65472.0
#define RAW_FAR_CLIP_THRESHOLD      1e-7

#ifndef kDieletricSpec
#define kDieletricSpec half4(0.04, 0.04, 0.04, 1.0 - 0.04)
#endif

// ============================================================================
// APV Support (Unity 2023.1+)
// ============================================================================

#if UNITY_VERSION >= 202310
#if defined(PROBE_VOLUMES_L1) || defined(PROBE_VOLUMES_L2)
#include "Packages/com.unity.render-pipelines.core/Runtime/Lighting/ProbeVolume/ProbeVolume.hlsl"

void SSGIEvaluateAdaptiveProbeVolume(in float3 posWS, in half3 normalWS, in half3 viewDir, in float2 positionSS, in uint renderingLayer,
    out half3 bakeDiffuseLighting, out half4 probeOcclusion)
{
    bakeDiffuseLighting = half3(0.0, 0.0, 0.0);

#if UNITY_VERSION >= 202330
    posWS = AddNoiseToSamplingPosition(posWS, positionSS, viewDir);
#else
    posWS = AddNoiseToSamplingPosition(posWS, positionSS);
#endif

#if UNITY_VERSION >= 600000
    APVSample apvSample = SampleAPV(posWS, normalWS, renderingLayer, viewDir);
#else
    APVSample apvSample = SampleAPV(posWS, normalWS, viewDir);
#endif
    
#ifdef USE_APV_PROBE_OCCLUSION
    probeOcclusion = apvSample.probeOcclusion;
#else
    probeOcclusion = 1;
#endif

    EvaluateAdaptiveProbeVolume(apvSample, normalWS, bakeDiffuseLighting);
}

#endif
#endif

// ============================================================================
// Ambient SH functions
// ============================================================================

void UpdateAmbientSH()
{
    unity_SHAr = ssgi_SHAr;
    unity_SHAg = ssgi_SHAg;
    unity_SHAb = ssgi_SHAb;
    unity_SHBr = ssgi_SHBr;
    unity_SHBg = ssgi_SHBg;
    unity_SHBb = ssgi_SHBb;
    unity_SHC = ssgi_SHC;
}

half3 SSGIEvaluateAmbientProbe(half3 normalWS)
{
    half3 res = SHEvalLinearL0L1(normalWS, ssgi_SHAr, ssgi_SHAg, ssgi_SHAb);
    res += SHEvalLinearL2(normalWS, ssgi_SHBr, ssgi_SHBg, ssgi_SHBb, ssgi_SHC);
    return res;
}

half3 SSGISampleProbeVolumePixel(in float3 absolutePositionWS, in float3 normalWS, in float3 viewDir, in float2 screenUV, out half4 probeOcclusion)
{
    probeOcclusion = 1.0;

#if defined(EVALUATE_SH_VERTEX) || defined(EVALUATE_SH_MIXED)
    return half3(0.0, 0.0, 0.0);
#elif defined(PROBE_VOLUMES_L1) || defined(PROBE_VOLUMES_L2)
    half3 bakedGI;
    if (_EnableProbeVolumes)
    {
        uint meshRenderingLayer = 0xFFFFFFFF;
        SSGIEvaluateAdaptiveProbeVolume(absolutePositionWS, normalWS, viewDir, screenUV * _ScreenSize.xy, meshRenderingLayer, bakedGI, probeOcclusion);
    }
    else
    {
        bakedGI = SSGIEvaluateAmbientProbe(normalWS);
    }
#ifdef UNITY_COLORSPACE_GAMMA
    bakedGI = LinearToSRGB(bakedGI);
#endif
    return bakedGI;
#else
    return half3(0, 0, 0);
#endif
}

half3 SSGIEvaluateAmbientProbeSRGB(half3 normalWS)
{
    half3 res = SSGIEvaluateAmbientProbe(normalWS);
#ifdef UNITY_COLORSPACE_GAMMA
    res = LinearToSRGB(res);
#endif
    return res;
}

// ============================================================================
// Ray and RayHit structures
// ============================================================================

struct Ray
{
    float3 position;
    half3  direction;
};

struct RayHit
{
    float3 position;
    float  distance;
    half3  normal;
    half3  emission;
};

RayHit InitializeRayHit()
{
    RayHit rayHit;
    rayHit.position = float3(0.0, 0.0, 0.0);
    rayHit.distance = REAL_EPS;
    rayHit.normal = half3(0.0, 0.0, 0.0);
    rayHit.emission = half3(0.0, 0.0, 0.0);
    return rayHit;
}

// ============================================================================
// Helper functions
// ============================================================================

float _InternalSeed;

float GenerateRandomValue(float2 screenUV)
{
    _InternalSeed += 1.0;
    return GenerateHashedRandomFloat(uint3(screenUV * _ScreenSize.xy, _FrameIndex + _InternalSeed));
}

half FastSign(half value)
{
    return value >= 0.0 ? 1.0 : -1.0;
}

float ConvertLinearEyeDepth(float deviceDepth)
{
    UNITY_BRANCH
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

half3 SampleHemisphereCosine(float u1, float u2, half3 normal)
{
    float phi = TWO_PI * u1;
    float cosTheta = sqrt(1.0 - u2);
    float sinTheta = sqrt(u2);
    
    float3 H = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    
    float3 up = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    
    return normalize(tangent * H.x + bitangent * H.y + normal * H.z);
}

// ============================================================================
// Upscaling functions
// ============================================================================

half3 DepthNormalsUpscale(float2 screenUV, float deviceDepth)
{
    float2 offsetUV = screenUV;
    offsetUV.y -= _IndirectDiffuseTexture_TexelSize.y;
    
    half3 centerNormal = SAMPLE_TEXTURE2D_X_LOD(_GBuffer2, sampler_PointClamp, screenUV, 0).xyz;
    float centerDepth = ConvertLinearEyeDepth(deviceDepth);
    
    float2 uv0 = offsetUV + float2(0.0, _IndirectDiffuseTexture_TexelSize.y);
    float2 uv1 = offsetUV + _IndirectDiffuseTexture_TexelSize.xy;
    float2 uv2 = offsetUV + float2(_IndirectDiffuseTexture_TexelSize.x, 0.0);
    float2 uv3 = offsetUV;
    
    float4 neighborDepth = float4(
        SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, sampler_PointClamp, uv0, 0).x,
        SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, sampler_PointClamp, uv1, 0).x,
        SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, sampler_PointClamp, uv2, 0).x,
        SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, sampler_PointClamp, uv3, 0).x);
    
#if !UNITY_REVERSED_Z
    neighborDepth = lerp(UNITY_NEAR_CLIP_VALUE.xxxx, float4(1.0, 1.0, 1.0, 1.0), neighborDepth);
#endif
    
    neighborDepth = float4(
        ConvertLinearEyeDepth(neighborDepth.x),
        ConvertLinearEyeDepth(neighborDepth.y),
        ConvertLinearEyeDepth(neighborDepth.z),
        ConvertLinearEyeDepth(neighborDepth.w));
    
    half3 normal0 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer2, sampler_PointClamp, uv0, 0).xyz;
    half3 normal1 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer2, sampler_PointClamp, uv1, 0).xyz;
    half3 normal2 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer2, sampler_PointClamp, uv2, 0).xyz;
    half3 normal3 = SAMPLE_TEXTURE2D_X_LOD(_GBuffer2, sampler_PointClamp, uv3, 0).xyz;
    
#if defined(_GBUFFER_NORMALS_OCT)
    half2 remappedOctNormalWS = half2(Unpack888ToFloat2(centerNormal));
    half2 octNormalWS = remappedOctNormalWS.xy * half(2.0) - half(1.0);
    centerNormal = half3(UnpackNormalOctQuadEncode(octNormalWS));
    
    remappedOctNormalWS = half2(Unpack888ToFloat2(normal0));
    octNormalWS = remappedOctNormalWS.xy * half(2.0) - half(1.0);
    normal0 = half3(UnpackNormalOctQuadEncode(octNormalWS));
    
    remappedOctNormalWS = half2(Unpack888ToFloat2(normal1));
    octNormalWS = remappedOctNormalWS.xy * half(2.0) - half(1.0);
    normal1 = half3(UnpackNormalOctQuadEncode(octNormalWS));
    
    remappedOctNormalWS = half2(Unpack888ToFloat2(normal2));
    octNormalWS = remappedOctNormalWS.xy * half(2.0) - half(1.0);
    normal2 = half3(UnpackNormalOctQuadEncode(octNormalWS));
    
    remappedOctNormalWS = half2(Unpack888ToFloat2(normal3));
    octNormalWS = remappedOctNormalWS.xy * half(2.0) - half(1.0);
    normal3 = half3(UnpackNormalOctQuadEncode(octNormalWS));
#endif
    
    half4 distances;
    distances.x = distance(neighborDepth.x, centerDepth);
    distances.y = distance(neighborDepth.y, centerDepth);
    distances.z = distance(neighborDepth.z, centerDepth);
    distances.w = distance(neighborDepth.w, centerDepth);
    
    distances.x *= (1 - saturate(dot(normal0, centerNormal)));
    distances.y *= (1 - saturate(dot(normal1, centerNormal)));
    distances.z *= (1 - saturate(dot(normal2, centerNormal)));
    distances.w *= (1 - saturate(dot(normal3, centerNormal)));
    
    half bestDistance = min(min(min(distances.x, distances.y), distances.z), distances.w);
    float2 bestUV = bestDistance == distances.x ? uv0 : 
                    bestDistance == distances.y ? uv1 : 
                    bestDistance == distances.z ? uv2 : uv3;
    
    return SAMPLE_TEXTURE2D_X_LOD(_IndirectDiffuseTexture, sampler_LinearClamp, bestUV, 0).xyz;
}

half3 DepthUpscale(float2 screenUV, float deviceDepth)
{
    float2 offsetUV = screenUV;
    offsetUV.y -= _IndirectDiffuseTexture_TexelSize.y;
    
    float centerDepth = Linear01Depth(deviceDepth, _ZBufferParams);
    
    float2 uv0 = offsetUV + float2(0.0, _IndirectDiffuseTexture_TexelSize.y);
    float2 uv1 = offsetUV + _IndirectDiffuseTexture_TexelSize.xy;
    float2 uv2 = offsetUV + float2(_IndirectDiffuseTexture_TexelSize.x, 0.0);
    float2 uv3 = offsetUV;
    
    float4 neighborDepth = float4(
        SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, sampler_PointClamp, uv0, 0).x,
        SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, sampler_PointClamp, uv1, 0).x,
        SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, sampler_PointClamp, uv2, 0).x,
        SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, sampler_PointClamp, uv3, 0).x);
    
#if !UNITY_REVERSED_Z
    neighborDepth = lerp(UNITY_NEAR_CLIP_VALUE.xxxx, float4(1.0, 1.0, 1.0, 1.0), neighborDepth);
#endif
    
    neighborDepth = float4(
        Linear01Depth(neighborDepth.x, _ZBufferParams),
        Linear01Depth(neighborDepth.y, _ZBufferParams),
        Linear01Depth(neighborDepth.z, _ZBufferParams),
        Linear01Depth(neighborDepth.w, _ZBufferParams));
    
    half4 distances;
    distances.x = abs(neighborDepth.x - centerDepth);
    distances.y = abs(neighborDepth.y - centerDepth);
    distances.z = abs(neighborDepth.z - centerDepth);
    distances.w = abs(neighborDepth.w - centerDepth);
    
    half bestDistance = min(min(min(distances.x, distances.y), distances.z), distances.w);
    float2 bestUV = bestDistance == distances.x ? uv0 : 
                    bestDistance == distances.y ? uv1 : 
                    bestDistance == distances.z ? uv2 : uv3;
    
    const half depthThreshold = 0.01;
    
    if (distances.x < depthThreshold && distances.y < depthThreshold && 
        distances.z < depthThreshold && distances.w < depthThreshold)
        return SAMPLE_TEXTURE2D_X_LOD(_IndirectDiffuseTexture, sampler_LinearClamp, bestUV, 0).xyz;
    else
        return SAMPLE_TEXTURE2D_X_LOD(_IndirectDiffuseTexture, sampler_PointClamp, screenUV, 0).xyz;
}

// ============================================================================
// Include fallback (reflection probes)
// ============================================================================

#include "./SSGIComputeFallback.hlsl"

#endif // SSGI_COMPUTE_UTILITIES_HLSL
