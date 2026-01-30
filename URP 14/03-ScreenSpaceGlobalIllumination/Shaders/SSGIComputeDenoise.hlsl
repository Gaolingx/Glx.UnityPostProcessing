// SSGIComputeDenoise.hlsl
#ifndef SSGI_COMPUTE_DENOISE_HLSL
#define SSGI_COMPUTE_DENOISE_HLSL

#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"

// ============================================================================
// Denoise configuration
// ============================================================================

#define BLUR_MAX_RADIUS 0.04
#define MIN_BLUR_DISTANCE 0.03
#define BLUR_OUT_RANGE 0.05

#define POISSON_SAMPLE_COUNT 8

static const half3 k_PoissonDiskSamples[POISSON_SAMPLE_COUNT] =
{
    half3(-1.00, 0.00, 1.0),
    half3(0.00, 1.00, 1.0),
    half3(1.00, 0.00, 1.0),
    half3(0.00, -1.00, 1.0),
    half3(-0.25 * sqrt(2.0), 0.25 * sqrt(2.0), 0.5),
    half3(0.25 * sqrt(2.0), 0.25 * sqrt(2.0), 0.5),
    half3(0.25 * sqrt(2.0), -0.25 * sqrt(2.0), 0.5),
    half3(-0.25 * sqrt(2.0), -0.25 * sqrt(2.0), 0.5)
};

half GetGaussianWeight(half r)
{
    return exp(-0.66 * r * r);
}

static const half k_GaussianWeight[POISSON_SAMPLE_COUNT] =
{
    GetGaussianWeight(k_PoissonDiskSamples[0].z),
    GetGaussianWeight(k_PoissonDiskSamples[1].z),
    GetGaussianWeight(k_PoissonDiskSamples[2].z),
    GetGaussianWeight(k_PoissonDiskSamples[3].z),
    GetGaussianWeight(k_PoissonDiskSamples[4].z),
    GetGaussianWeight(k_PoissonDiskSamples[5].z),
    GetGaussianWeight(k_PoissonDiskSamples[6].z),
    GetGaussianWeight(k_PoissonDiskSamples[7].z)
};

// ============================================================================
// Helper functions
// ============================================================================

half sqr(half value)
{
    return value * value;
}

half ComputeMaxReprojectionWorldRadius(float3 positionWS, half3 viewDirWS, half3 normalWS, 
    half pixelSpreadAngleTangent, half maxDistance, half pixelTolerance)
{
    half parallelPixelFootPrint = pixelSpreadAngleTangent * length(positionWS - GetCameraPositionWS());
    half realPixelFootPrint = parallelPixelFootPrint / max(abs(dot(normalWS, viewDirWS)), PROJECTION_EPSILON);
    return max(maxDistance, realPixelFootPrint * pixelTolerance);
}

half ComputeMaxReprojectionWorldRadius(float3 positionWS, half3 viewDirWS, half3 normalWS, half pixelSpreadAngleTangent)
{
    return ComputeMaxReprojectionWorldRadius(positionWS, viewDirWS, normalWS, pixelSpreadAngleTangent, 
        MAX_REPROJECTION_DISTANCE, MAX_PIXEL_TOLERANCE);
}

// ============================================================================
// Color box adjustment for temporal reprojection
// ============================================================================

void AdjustColorBox(inout half3 boxMin, inout half3 boxMax, inout half3 moment1, inout half3 moment2, 
    float2 uv, half currX, half currY, Texture2D<float4> tex, SamplerState samp, float2 texelSize)
{
    float2 sampleUV = uv + float2(currX, currY) * texelSize;
    half3 color = SAMPLE_TEXTURE2D_X_LOD(tex, samp, sampleUV, 0).xyz;
    boxMin = min(color, boxMin);
    boxMax = max(color, boxMax);
    moment1 += color;
    moment2 += color * color;
}

half3 DirectClipToAABB(half3 history, half3 minimum, half3 maximum)
{
    half3 center = 0.5 * (maximum + minimum);
    half3 extents = 0.5 * (maximum - minimum);
    half3 offset = history - center;
    half3 v_unit = offset.xyz / extents.xyz;
    half3 absUnit = abs(v_unit);
    half maxUnit = Max3(absUnit.x, absUnit.y, absUnit.z);
    if (maxUnit > 1.0)
        return center + (offset / maxUnit);
    else
        return history;
}

// ============================================================================
// Specular dominant direction functions
// ============================================================================

half GetSpecularDominantFactor(half NoV, half linearRoughness)
{
    half a = 0.298475 * log(39.4115 - 39.0029 * linearRoughness);
    half dominantFactor = pow(saturate(1.0 - NoV), 10.8649) * (1.0 - a) + a;
    return saturate(dominantFactor);
}

half GetSpecularDominantFactor(half NoV)
{
    half a = 0.298475 * log(39.4115 - 39.0029 * 1.0);
    half dominantFactor = pow(saturate(1.0 - NoV), 10.8649) * (1.0 - a) + a;
    return saturate(dominantFactor);
}

half3 GetSpecularDominantDirectionWithFactor(half3 N, half3 V, half dominantFactor)
{
    half3 R = reflect(-V, N);
    half3 D = lerp(N, R, dominantFactor);
    return normalize(D);
}

half4 GetSpecularDominantDirection(half3 N, half3 V, half linearRoughness)
{
    half NoV = abs(dot(N, V));
    half dominantFactor = GetSpecularDominantFactor(NoV, linearRoughness);
    return half4(GetSpecularDominantDirectionWithFactor(N, V, dominantFactor), dominantFactor);
}

half4 GetSpecularDominantDirection(half3 N, half3 V)
{
    half NoV = abs(dot(N, V));
    half dominantFactor = GetSpecularDominantFactor(NoV);
    return half4(GetSpecularDominantDirectionWithFactor(N, V, dominantFactor), dominantFactor);
}

// ============================================================================
// Kernel basis functions
// ============================================================================

half2x3 GetKernelBasis(half3 V, half3 N, half linearRoughness)
{
    half3x3 basis = GetLocalFrame(N);
    half3 T = basis[0];
    half3 B = basis[1];
    half NoV = abs(dot(N, V));
    half f = GetSpecularDominantFactor(NoV, linearRoughness);
    half3 R = reflect(-V, N);
    half3 D = normalize(lerp(N, R, f));
    half NoD = abs(dot(N, D));
    
    if (NoD < 0.999 && linearRoughness != 1.0)
    {
        half3 Dreflected = reflect(-D, N);
        T = normalize(cross(N, Dreflected));
        B = cross(Dreflected, T);
        
        half acos01sq = saturate(1.0 - NoV);
        half skewFactor = lerp(1.0, linearRoughness, sqrt(acos01sq));
        T *= skewFactor;
    }
    
    return half2x3(T, B);
}

half2x3 GetKernelBasis(half3 V, half3 N)
{
    half3x3 basis = GetLocalFrame(N);
    half3 T = basis[0];
    half3 B = basis[1];
    return half2x3(T, B);
}

half2 RotateVector(half4 rotator, half2 v)
{
    return v.x * rotator.xz + v.y * rotator.yw;
}

float2 GetKernelSampleCoordinates(half3 offset, float3 X, half3 T, half3 B, half4 rotator)
{
    offset.xy = RotateVector(rotator, offset.xy);
    float3 wsPos = X + T * offset.x + B * offset.y;
    float4 hClip = TransformWorldToHClip(wsPos);
    hClip.xyz /= hClip.w;
    float2 nDC = hClip.xy * 0.5 + 0.5;
#if UNITY_UV_STARTS_AT_TOP
    nDC.y = 1.0 - nDC.y;
#endif
    return nDC;
}

#endif // SSGI_COMPUTE_DENOISE_HLSL
