// SSGIComputeInput.hlsl
#ifndef SSGI_COMPUTE_INPUT_HLSL
#define SSGI_COMPUTE_INPUT_HLSL

#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
#include "./Core.hlsl"

// ============================================================================
// Material flags (from URP GBuffer)
// ============================================================================

#define kLightingInvalid  -1
#define kLightingLit       1
#define kLightingSimpleLit 2
#define kLightFlagSubtractiveMixedLighting    4

#define kMaterialFlagReceiveShadowsOff        1
#define kMaterialFlagSpecularHighlightsOff    2
#define kMaterialFlagSubtractiveMixedLighting 4
#define kMaterialFlagSpecularSetup            8

// ============================================================================
// Texture inputs (Read-only)
// ============================================================================

TEXTURE2D_X(_SourceTexture);
TEXTURE2D_X(_SourceDiffuseTexture);

TEXTURE2D_X_HALF(_GBuffer0);
TEXTURE2D_X_HALF(_GBuffer1);
TEXTURE2D_X_HALF(_GBuffer2);

TEXTURE2D_X_FLOAT(_CameraBackDepthTexture);
TEXTURE2D_X_HALF(_CameraBackOpaqueTexture);

TEXTURE2D_X(_MotionVectorTexture);
TEXTURE2D_X(_HistoryIndirectDiffuseTexture);
TEXTURE2D_X(_SSGISampleTexture);
TEXTURE2D_X(_SSGIHistorySampleTexture);
TEXTURE2D_X_FLOAT(_SSGIHistoryDepthTexture);
TEXTURE2D_X(_IndirectDiffuseTexture);
TEXTURE2D_X(_IntermediateIndirectDiffuseTexture);
TEXTURE2D_X(_SSGIHistoryCameraColorTexture);
TEXTURE2D_X(_IntermediateCameraColorTexture);
TEXTURE2D_X(_APVLightingTexture);

// ============================================================================
// Texture outputs (Read-Write)
// ============================================================================

RW_TEXTURE2D_X(float4, _RW_CameraColorTexture);
RW_TEXTURE2D_X(float4, _RW_IntermediateCameraColorTexture);
RW_TEXTURE2D_X(float4, _RW_IndirectDiffuseTexture);
RW_TEXTURE2D_X(float4, _RW_IntermediateIndirectDiffuseTexture);
RW_TEXTURE2D_X(float, _RW_SSGIHistoryDepthTexture);
RW_TEXTURE2D_X(float, _RW_SSGISampleTexture);
RW_TEXTURE2D_X(float4, _RW_HistoryIndirectDiffuseTexture);
RW_TEXTURE2D_X(float, _RW_SSGIHistorySampleTexture);
RW_TEXTURE2D_X(float4, _RW_APVLightingTexture);
RW_TEXTURE2D_X(float4, _RW_MotionVectorTexture);
RW_TEXTURE2D_X(float4, _RW_DestinationTexture);

// ============================================================================
// Constant buffer - matches C# property names
// ============================================================================

CBUFFER_START(SSGIComputeParams)
    // Ray marching parameters
    float _MaxSteps;
    float _MaxSmallSteps;
    float _MaxMediumSteps;
    float _StepSize;
    float _SmallStepSize;
    float _MediumStepSize;
    float _Thickness;
    float _Thickness_Increment;
    float _RayCount;
    
    // Denoising parameters
    float _TemporalIntensity;
    float _AggressiveDenoise;
    float4 _ReBlurBlurRotator;
    float _ReBlurDenoiserRadius;
    
    // Quality and rendering
    float _MaxBrightness;
    float _IsProbeCamera;
    float _BackDepthEnabled;
    float _DownSample;
    float _FrameIndex;
    float _HistoryTextureValid;
    float _PixelSpreadAngleTangent;
    
    // Lighting
    float _IndirectDiffuseLightingMultiplier;
    uint _IndirectDiffuseRenderingLayers;
    
    // Screen and texture sizes
    float4 _SSGITextureSize;               // (width, height, 1/width, 1/height) - may be different from screen if downsampled
    float4 _IndirectDiffuseTexture_TexelSize;
    float4 _MotionVectorTexture_TexelSize;
    
    // History matrices
    float4x4 _PrevInvViewProjMatrix;
    float3 _PrevCameraPositionWS;
    float _Padding0;
    
    // Internal seed for random generation
    float _Seed;
CBUFFER_END

// ============================================================================
// Ambient SH coefficients
// ============================================================================

half4 ssgi_SHAr;
half4 ssgi_SHAg;
half4 ssgi_SHAb;
half4 ssgi_SHBr;
half4 ssgi_SHBg;
half4 ssgi_SHBb;
half4 ssgi_SHC;

// ============================================================================
// Reflection Probe parameters (non Forward+)
// ============================================================================

#ifndef _FP_REFL_PROBE_ATLAS
TEXTURECUBE(_SpecCube0);
SAMPLER(sampler_SpecCube0);
float4 _SpecCube0_ProbePosition;
float3 _SpecCube0_BoxMin;
float3 _SpecCube0_BoxMax;
half4 _SpecCube0_HDR;
TEXTURECUBE(_SpecCube1);
SAMPLER(sampler_SpecCube1);
float4 _SpecCube1_ProbePosition;
float3 _SpecCube1_BoxMin;
float3 _SpecCube1_BoxMax;
half4 _SpecCube1_HDR;
half _ProbeWeight;
half _ProbeSet;
#endif

// ============================================================================
// Helper macros for XR
// ============================================================================

#if defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_STEREO_MULTIVIEW_ENABLED)
    #define COORD_TEXTURE2D_X(pixelCoord) uint3(pixelCoord, unity_StereoEyeIndex)
    #define UNITY_XR_ASSIGN_VIEW_INDEX(viewIndex) unity_StereoEyeIndex = viewIndex
#else
    #define COORD_TEXTURE2D_X(pixelCoord) pixelCoord
    #define UNITY_XR_ASSIGN_VIEW_INDEX(viewIndex)
#endif

#endif // SSGI_COMPUTE_INPUT_HLSL
