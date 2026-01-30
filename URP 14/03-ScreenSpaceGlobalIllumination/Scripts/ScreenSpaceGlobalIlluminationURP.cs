// ScreenSpaceGlobalIlluminationURP.cs
using System;
using System.Reflection;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RendererUtils;
using UnityEngine.Experimental.Rendering;

#if UNITY_6000_0_OR_NEWER
using UnityEngine.Rendering.RenderGraphModule;
#endif

[DisallowMultipleRendererFeature("Screen Space Global Illumination")]
[Tooltip("The Screen Space Global Illumination uses the depth and color buffer of the screen to calculate diffuse light bounces.")]
[HelpURL("https://github.com/jiaozi158/UnitySSGIURP/blob/main")]
public class ScreenSpaceGlobalIlluminationURP : ScriptableRendererFeature
{
    [Header("Setup")]
    [Tooltip("The compute shader for screen space global illumination.")]
    [SerializeField] private ComputeShader m_ComputeShader;
    
    [Tooltip("Specifies if URP computes screen space global illumination in Rendering Debugger view.")]
    [SerializeField] private bool m_RenderingDebugger = false;

    [Header("Performance")]
    [Tooltip("Specifies if URP computes screen space global illumination in both real-time and baked reflection probes.")]
    [SerializeField] private bool m_ReflectionProbes = true;
    
    [Tooltip("Enables high-quality upscaling for screen space global illumination.")]
    [SerializeField] private bool m_HighQualityUpscaling = false;

    [Header("Lighting")]
    [Tooltip("Specifies if screen space global illumination overrides ambient lighting.")]
    [SerializeField] private bool m_OverrideAmbientLighting = true;

    [Header("Advanced")]
    [Tooltip("Renders back-face lighting when using automatic thickness mode.")]
    [SerializeField] private bool m_BackfaceLighting = false;

    /// <summary>
    /// Gets or sets the compute shader for SSGI.
    /// </summary>
    public ComputeShader SSGIComputeShader
    {
        get => m_ComputeShader;
        set => m_ComputeShader = value;
    }

    /// <summary>
    /// Gets or sets whether to render in Rendering Debugger view.
    /// </summary>
    public bool RenderingDebugger
    {
        get => m_RenderingDebugger;
        set => m_RenderingDebugger = value;
    }

    /// <summary>
    /// Gets or sets whether to compute SSGI in reflection probes.
    /// </summary>
    public bool ReflectionProbes
    {
        get => m_ReflectionProbes;
        set => m_ReflectionProbes = value;
    }

    /// <summary>
    /// Gets or sets whether to enable high-quality upscaling.
    /// </summary>
    public bool HighQualityUpscaling
    {
        get => m_HighQualityUpscaling;
        set => m_HighQualityUpscaling = value;
    }

    /// <summary>
    /// Gets or sets whether SSGI overrides ambient lighting.
    /// </summary>
    public bool OverrideAmbientLighting
    {
        get => m_OverrideAmbientLighting;
        set => m_OverrideAmbientLighting = value;
    }

    /// <summary>
    /// Gets or sets whether to render backface lighting.
    /// </summary>
    public bool BackfaceLighting
    {
        get => m_BackfaceLighting;
        set => m_BackfaceLighting = value;
    }

    // Render Passes
    private PreRenderScreenSpaceGlobalIlluminationPass m_PreRenderSSGIPass;
    private ScreenSpaceGlobalIlluminationPass m_SSGIPass;
    private BackfaceDataPass m_BackfaceDataPass;
    private ForwardGBufferPass m_ForwardGBufferPass;

    // GBuffer pass names
    private readonly string[] m_GBufferPassNames = new string[] { "UniversalGBuffer" };

    // Reflection to get URP internal fields
    private static readonly FieldInfo gBufferFieldInfo = typeof(UniversalRenderer).GetField("m_GBufferPass", BindingFlags.NonPublic | BindingFlags.Instance);
    private static readonly FieldInfo motionVectorPassFieldInfo = typeof(UniversalRenderer).GetField("m_MotionVectorPass", BindingFlags.NonPublic | BindingFlags.Instance);

    // Logging flags
    private bool isComputeShaderMissingLogPrinted = false;
    private bool isDebuggerLogPrinted = false;
    private bool isBackfaceLightingLogPrinted = false;

    #region Shader Property IDs
    // Compute shader kernel indices
    private static class KernelIndices
    {
        public const int CopyDirectLighting = 0;
        public const int SSGIRayMarching = 1;
        public const int TemporalReprojection = 2;
        public const int SpatialDenoise = 3;
        public const int TemporalStabilization = 4;
        public const int CopyHistoryDepth = 5;
        public const int CombineGI = 6;
        public const int CameraMotionVectors = 7;
        public const int PoissonDiskDenoise = 8;
        public const int BlitColorTexture = 9;
    }

    // Shader property IDs
    private static readonly int _MaxSteps = Shader.PropertyToID("_MaxSteps");
    private static readonly int _MaxSmallSteps = Shader.PropertyToID("_MaxSmallSteps");
    private static readonly int _MaxMediumSteps = Shader.PropertyToID("_MaxMediumSteps");
    private static readonly int _Thickness = Shader.PropertyToID("_Thickness");
    private static readonly int _Thickness_Increment = Shader.PropertyToID("_Thickness_Increment");
    private static readonly int _StepSize = Shader.PropertyToID("_StepSize");
    private static readonly int _SmallStepSize = Shader.PropertyToID("_SmallStepSize");
    private static readonly int _MediumStepSize = Shader.PropertyToID("_MediumStepSize");
    private static readonly int _RayCount = Shader.PropertyToID("_RayCount");
    private static readonly int _TemporalIntensity = Shader.PropertyToID("_TemporalIntensity");
    private static readonly int _MaxBrightness = Shader.PropertyToID("_MaxBrightness");
    private static readonly int _IsProbeCamera = Shader.PropertyToID("_IsProbeCamera");
    private static readonly int _BackDepthEnabled = Shader.PropertyToID("_BackDepthEnabled");
    private static readonly int _PrevInvViewProjMatrix = Shader.PropertyToID("_PrevInvViewProjMatrix");
    private static readonly int _PrevCameraPositionWS = Shader.PropertyToID("_PrevCameraPositionWS");
    private static readonly int _PixelSpreadAngleTangent = Shader.PropertyToID("_PixelSpreadAngleTangent");
    private static readonly int _HistoryTextureValid = Shader.PropertyToID("_HistoryTextureValid");
    private static readonly int _IndirectDiffuseLightingMultiplier = Shader.PropertyToID("_IndirectDiffuseLightingMultiplier");
    private static readonly int _IndirectDiffuseRenderingLayers = Shader.PropertyToID("_IndirectDiffuseRenderingLayers");
    private static readonly int _AggressiveDenoise = Shader.PropertyToID("_AggressiveDenoise");
    private static readonly int _ReBlurBlurRotator = Shader.PropertyToID("_ReBlurBlurRotator");
    private static readonly int _ReBlurDenoiserRadius = Shader.PropertyToID("_ReBlurDenoiserRadius");
    private static readonly int _DownSample = Shader.PropertyToID("_DownSample");
    private static readonly int _FrameIndex = Shader.PropertyToID("_FrameIndex");
    private static readonly int _Seed = Shader.PropertyToID("_Seed");
    private static readonly int _ScreenSize = Shader.PropertyToID("_ScreenSize");
    private static readonly int _SSGITextureSize = Shader.PropertyToID("_SSGITextureSize");

    // Texture property IDs
    private static readonly int _SourceTexture = Shader.PropertyToID("_SourceTexture");
    private static readonly int _SourceDiffuseTexture = Shader.PropertyToID("_SourceDiffuseTexture");
    private static readonly int _CameraDepthTexture = Shader.PropertyToID("_CameraDepthTexture");
    private static readonly int _GBuffer0 = Shader.PropertyToID("_GBuffer0");
    private static readonly int _GBuffer1 = Shader.PropertyToID("_GBuffer1");
    private static readonly int _GBuffer2 = Shader.PropertyToID("_GBuffer2");
    private static readonly int _MotionVectorTexture = Shader.PropertyToID("_MotionVectorTexture");
    private static readonly int _CameraBackDepthTexture = Shader.PropertyToID("_CameraBackDepthTexture");
    private static readonly int _CameraBackOpaqueTexture = Shader.PropertyToID("_CameraBackOpaqueTexture");

    // RW Texture property IDs
    private static readonly int _RW_CameraColorTexture = Shader.PropertyToID("_RW_CameraColorTexture");
    private static readonly int _RW_IntermediateCameraColorTexture = Shader.PropertyToID("_RW_IntermediateCameraColorTexture");
    private static readonly int _RW_IndirectDiffuseTexture = Shader.PropertyToID("_RW_IndirectDiffuseTexture");
    private static readonly int _RW_IntermediateIndirectDiffuseTexture = Shader.PropertyToID("_RW_IntermediateIndirectDiffuseTexture");
    private static readonly int _RW_SSGIHistoryDepthTexture = Shader.PropertyToID("_RW_SSGIHistoryDepthTexture");
    private static readonly int _RW_SSGISampleTexture = Shader.PropertyToID("_RW_SSGISampleTexture");
    private static readonly int _RW_HistoryIndirectDiffuseTexture = Shader.PropertyToID("_RW_HistoryIndirectDiffuseTexture");
    private static readonly int _RW_SSGIHistorySampleTexture = Shader.PropertyToID("_RW_SSGIHistorySampleTexture");
    private static readonly int _RW_APVLightingTexture = Shader.PropertyToID("_RW_APVLightingTexture");
    private static readonly int _RW_MotionVectorTexture = Shader.PropertyToID("_RW_MotionVectorTexture");
    private static readonly int _RW_DestinationTexture = Shader.PropertyToID("_RW_DestinationTexture");

    // History texture property IDs (read-only)
    private static readonly int _IndirectDiffuseTexture = Shader.PropertyToID("_IndirectDiffuseTexture");
    private static readonly int _IntermediateIndirectDiffuseTexture = Shader.PropertyToID("_IntermediateIndirectDiffuseTexture");
    private static readonly int _IntermediateCameraColorTexture = Shader.PropertyToID("_IntermediateCameraColorTexture");
    private static readonly int _HistoryIndirectDiffuseTexture = Shader.PropertyToID("_HistoryIndirectDiffuseTexture");
    private static readonly int _SSGIHistoryDepthTexture = Shader.PropertyToID("_SSGIHistoryDepthTexture");
    private static readonly int _SSGISampleTexture = Shader.PropertyToID("_SSGISampleTexture");
    private static readonly int _SSGIHistorySampleTexture = Shader.PropertyToID("_SSGIHistorySampleTexture");
    private static readonly int _SSGIHistoryCameraColorTexture = Shader.PropertyToID("_SSGIHistoryCameraColorTexture");
    private static readonly int _APVLightingTexture = Shader.PropertyToID("_APVLightingTexture");
    private static readonly int _IndirectDiffuseTexture_TexelSize = Shader.PropertyToID("_IndirectDiffuseTexture_TexelSize");

    // Reflection probe property IDs
    private static readonly int _SpecCube0 = Shader.PropertyToID("_SpecCube0");
    private static readonly int _SpecCube0_HDR = Shader.PropertyToID("_SpecCube0_HDR");
    private static readonly int _SpecCube0_BoxMin = Shader.PropertyToID("_SpecCube0_BoxMin");
    private static readonly int _SpecCube0_BoxMax = Shader.PropertyToID("_SpecCube0_BoxMax");
    private static readonly int _SpecCube0_ProbePosition = Shader.PropertyToID("_SpecCube0_ProbePosition");
    private static readonly int _ProbeWeight = Shader.PropertyToID("_ProbeWeight");
    private static readonly int _ProbeSet = Shader.PropertyToID("_ProbeSet");

    // Ambient SH property IDs
    private static readonly int ssgi_SHAr = Shader.PropertyToID("ssgi_SHAr");
    private static readonly int ssgi_SHAg = Shader.PropertyToID("ssgi_SHAg");
    private static readonly int ssgi_SHAb = Shader.PropertyToID("ssgi_SHAb");
    private static readonly int ssgi_SHBr = Shader.PropertyToID("ssgi_SHBr");
    private static readonly int ssgi_SHBg = Shader.PropertyToID("ssgi_SHBg");
    private static readonly int ssgi_SHBb = Shader.PropertyToID("ssgi_SHBb");
    private static readonly int ssgi_SHC = Shader.PropertyToID("ssgi_SHC");

    // Matrix property IDs (for motion vectors)
    private static readonly int _PrevViewProjMatrix = Shader.PropertyToID("_PrevViewProjMatrix");
    private static readonly int _NonJitteredViewProjMatrix = Shader.PropertyToID("_NonJitteredViewProjMatrix");
    #endregion

    #region Keyword Names
    private const string _FP_REFL_PROBE_ATLAS = "_FP_REFL_PROBE_ATLAS";
    private const string _RAYMARCHING_FALLBACK_SKY = "_RAYMARCHING_FALLBACK_SKY";
    private const string _RAYMARCHING_FALLBACK_REFLECTION_PROBES = "_RAYMARCHING_FALLBACK_REFLECTION_PROBES";
    private const string _BACKFACE_TEXTURES = "_BACKFACE_TEXTURES";
    private const string _FORWARD_PLUS = "_FORWARD_PLUS";
    private const string _GBUFFER_NORMALS_OCT = "_GBUFFER_NORMALS_OCT";
    private const string _WRITE_RENDERING_LAYERS = "_WRITE_RENDERING_LAYERS";
    private const string _USE_RENDERING_LAYERS = "_USE_RENDERING_LAYERS";
    private const string _DEPTH_NORMALS_UPSCALE = "_DEPTH_NORMALS_UPSCALE";
    private const string PROBE_VOLUMES_L1 = "PROBE_VOLUMES_L1";
    private const string PROBE_VOLUMES_L2 = "PROBE_VOLUMES_L2";
    private const string _APV_LIGHTING_BUFFER = "_APV_LIGHTING_BUFFER";

#if UNITY_6000_1_OR_NEWER
    private const string _CLUSTER_LIGHT_LOOP = "_CLUSTER_LIGHT_LOOP";
    private const string _REFLECTION_PROBE_ATLAS = "_REFLECTION_PROBE_ATLAS";
#endif

    // Global keywords
    private const string SSGI_RENDER_GBUFFER = "SSGI_RENDER_GBUFFER";
    private const string SSGI_RENDER_BACKFACE_DEPTH = "SSGI_RENDER_BACKFACE_DEPTH";
    private const string SSGI_RENDER_BACKFACE_COLOR = "SSGI_RENDER_BACKFACE_COLOR";
    #endregion

    // Denoise configuration
    private const float k_BlurMaxRadius = 0.04f;

    public override void Create()
    {
        if (m_ComputeShader == null)
        {
#if UNITY_EDITOR || DEBUG
            if (!isComputeShaderMissingLogPrinted)
            {
                Debug.LogError("Screen Space Global Illumination URP: Compute Shader is not assigned.");
                isComputeShaderMissingLogPrinted = true;
            }
#endif
            return;
        }
        else
        {
            isComputeShaderMissingLogPrinted = false;
        }

        if (m_PreRenderSSGIPass == null)
        {
            m_PreRenderSSGIPass = new PreRenderScreenSpaceGlobalIlluminationPass();
#if UNITY_6000_0_OR_NEWER
            m_PreRenderSSGIPass.renderPassEvent = RenderPassEvent.BeforeRenderingPrePasses;
#else
            m_PreRenderSSGIPass.renderPassEvent = RenderPassEvent.BeforeRenderingTransparents - 1;
#endif
        }

        if (m_SSGIPass == null)
        {
            m_SSGIPass = new ScreenSpaceGlobalIlluminationPass();
#if UNITY_6000_0_OR_NEWER
            bool enableRenderGraph = !GraphicsSettings.GetRenderPipelineSettings<RenderGraphSettings>().enableRenderCompatibilityMode;
            m_SSGIPass.renderPassEvent = enableRenderGraph ? RenderPassEvent.AfterRenderingSkybox : RenderPassEvent.BeforeRenderingTransparents;
#else
            m_SSGIPass.renderPassEvent = RenderPassEvent.BeforeRenderingTransparents;
#endif
        }

        if (m_BackfaceDataPass == null)
        {
            m_BackfaceDataPass = new BackfaceDataPass();
            m_BackfaceDataPass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques - 1;
        }

        if (m_ForwardGBufferPass == null)
        {
            m_ForwardGBufferPass = new ForwardGBufferPass(m_GBufferPassNames);
            m_ForwardGBufferPass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;
        }
    }

    protected override void Dispose(bool disposing)
    {
        m_PreRenderSSGIPass?.Dispose();
        m_SSGIPass?.Dispose();
        m_BackfaceDataPass?.Dispose();
        m_ForwardGBufferPass?.Dispose();
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        if (isComputeShaderMissingLogPrinted || m_ComputeShader == null)
            return;

        if (renderingData.cameraData.camera.cameraType == CameraType.Preview)
            return;

        var stack = VolumeManager.instance.stack;
        ScreenSpaceGlobalIlluminationVolume ssgiVolume = stack.GetComponent<ScreenSpaceGlobalIlluminationVolume>();
        
        if (ssgiVolume == null || !ssgiVolume.IsActive())
            return;

        bool isDebugger = DebugManager.instance.isAnyDebugUIActive;
        bool shouldDisable = !m_ReflectionProbes && renderingData.cameraData.camera.cameraType == CameraType.Reflection;
        shouldDisable |= ssgiVolume.indirectDiffuseLightingMultiplier.value == 0.0f && !m_OverrideAmbientLighting;
        shouldDisable |= renderingData.cameraData.renderType == CameraRenderType.Overlay;

        if (shouldDisable)
            return;

#if UNITY_EDITOR || DEBUG
        if (isDebugger && !m_RenderingDebugger)
        {
            if (!isDebuggerLogPrinted)
            {
                Debug.Log("Screen Space Global Illumination URP: Disable effect to avoid affecting rendering debugging.");
                isDebuggerLogPrinted = true;
            }
            return;
        }
        else
        {
            isDebuggerLogPrinted = false;
        }
#else
        if (isDebugger && !m_RenderingDebugger)
            return;
#endif

        // Configure compute shader keywords and parameters
        ConfigureComputeShader(ssgiVolume, renderingData);

        // Configure pass settings
        m_SSGIPass.computeShader = m_ComputeShader;
        m_SSGIPass.ssgiVolume = ssgiVolume;
        m_SSGIPass.overrideAmbientLighting = m_OverrideAmbientLighting;
        m_SSGIPass.highQualityUpscaling = m_HighQualityUpscaling;

#if UNITY_EDITOR
        // Scene view motion vector fix
        if (renderingData.cameraData.camera.cameraType == CameraType.SceneView)
        {
            m_PreRenderSSGIPass.computeShader = m_ComputeShader;
            renderer.EnqueuePass(m_PreRenderSSGIPass);
        }
#endif

        if (renderingData.cameraData.camera.cameraType != CameraType.Preview && (!isDebugger || m_RenderingDebugger))
            renderer.EnqueuePass(m_SSGIPass);

        // Check rendering path
        bool isUsingDeferred = gBufferFieldInfo.GetValue(renderer) != null;
        isUsingDeferred &= (SystemInfo.graphicsDeviceType != GraphicsDeviceType.OpenGLES3) && 
                           (SystemInfo.graphicsDeviceType != GraphicsDeviceType.OpenGLCore);

        // Backface data pass
        bool renderBackfaceData = ssgiVolume.thicknessMode.value != ScreenSpaceGlobalIlluminationVolume.ThicknessMode.Constant;
        if (renderBackfaceData)
        {
            bool supportBackfaceLighting = m_BackfaceLighting && !isUsingDeferred;
            m_BackfaceDataPass.backfaceLighting = supportBackfaceLighting;
            renderer.EnqueuePass(m_BackfaceDataPass);

            Shader.EnableKeyword(SSGI_RENDER_BACKFACE_DEPTH);
            if (supportBackfaceLighting)
            {
                Shader.EnableKeyword(SSGI_RENDER_BACKFACE_COLOR);
            }
            else
            {
                Shader.DisableKeyword(SSGI_RENDER_BACKFACE_COLOR);
            }
        }
        else
        {
            Shader.DisableKeyword(SSGI_RENDER_BACKFACE_DEPTH);
            Shader.DisableKeyword(SSGI_RENDER_BACKFACE_COLOR);
        }

#if UNITY_EDITOR || DEBUG
        if (m_BackfaceLighting && isUsingDeferred)
        {
            if (!isBackfaceLightingLogPrinted)
            {
                Debug.LogError("Screen Space Global Illumination URP: Backface Lighting is only supported on Forward(+) rendering path.");
                isBackfaceLightingLogPrinted = true;
            }
        }
        else
        {
            isBackfaceLightingLogPrinted = false;
        }
#endif

        // Forward GBuffer pass
        if (!isUsingDeferred)
        {
            renderer.EnqueuePass(m_ForwardGBufferPass);
            Shader.EnableKeyword(SSGI_RENDER_GBUFFER);
        }
        else
        {
            Shader.DisableKeyword(SSGI_RENDER_GBUFFER);
        }
    }

    private void ConfigureComputeShader(ScreenSpaceGlobalIlluminationVolume ssgiVolume, RenderingData renderingData)
    {
        // Ray marching quality settings
        bool lowStepCount = ssgiVolume.maxRaySteps.value <= 16;
        int groupsCount = ssgiVolume.maxRaySteps.value / 8;
        int smallSteps = lowStepCount ? 0 : Mathf.Max(groupsCount, 4);
        int mediumSteps = lowStepCount ? groupsCount + 2 : smallSteps + groupsCount * 2;

        float resolutionScale = ssgiVolume.fullResolutionSS.value ? 1.0f : ssgiVolume.resolutionScaleSS.value;
        float temporalIntensity = Mathf.Lerp(ssgiVolume.denoiseIntensitySS.value + 0.02f, 
                                              ssgiVolume.denoiseIntensitySS.value - 0.04f, resolutionScale);

        // Set compute shader parameters
        m_ComputeShader.SetFloat(_MaxSteps, ssgiVolume.maxRaySteps.value);
        m_ComputeShader.SetFloat(_MaxSmallSteps, smallSteps);
        m_ComputeShader.SetFloat(_MaxMediumSteps, mediumSteps);
        m_ComputeShader.SetFloat(_StepSize, lowStepCount ? 0.5f : 0.4f);
        m_ComputeShader.SetFloat(_SmallStepSize, smallSteps < 4 ? 0.05f : 0.015f);
        m_ComputeShader.SetFloat(_MediumStepSize, lowStepCount ? 0.1f : 0.05f);
        m_ComputeShader.SetFloat(_Thickness, ssgiVolume.depthBufferThickness.value);
        m_ComputeShader.SetFloat(_Thickness_Increment, ssgiVolume.depthBufferThickness.value * 0.25f);
        m_ComputeShader.SetFloat(_RayCount, ssgiVolume.sampleCount.value);
        m_ComputeShader.SetFloat(_TemporalIntensity, temporalIntensity);
        m_ComputeShader.SetFloat(_ReBlurDenoiserRadius, ssgiVolume.denoiserRadiusSS.value * 2.0f * k_BlurMaxRadius);
        m_ComputeShader.SetFloat(_IndirectDiffuseLightingMultiplier, ssgiVolume.indirectDiffuseLightingMultiplier.value);
        m_ComputeShader.SetFloat(_MaxBrightness, 7.0f);
        m_ComputeShader.SetFloat(_AggressiveDenoise, 
            ssgiVolume.denoiserAlgorithmSS.value == ScreenSpaceGlobalIlluminationVolume.DenoiserAlgorithm.Aggressive ? 1.0f : 0.0f);

        // Rendering layers
#if UNITY_2023_3_OR_NEWER
        bool enableRenderingLayers = Shader.IsKeywordEnabled(_WRITE_RENDERING_LAYERS) && 
                                      ssgiVolume.indirectDiffuseRenderingLayers.value.value != 0xFFFF;
        if (enableRenderingLayers)
        {
            EnableKeyword(_USE_RENDERING_LAYERS);
            m_ComputeShader.SetInt(_IndirectDiffuseRenderingLayers, (int)ssgiVolume.indirectDiffuseRenderingLayers.value.value);
        }
        else
        {
            DisableKeyword(_USE_RENDERING_LAYERS);
        }
        m_SSGIPass.enableRenderingLayers = enableRenderingLayers;
#else
        DisableKeyword(_USE_RENDERING_LAYERS);
        m_SSGIPass.enableRenderingLayers = false;
#endif

        // Fallback settings
        bool skyFallback = ssgiVolume.IsFallbackSky();
        SetKeyword(_RAYMARCHING_FALLBACK_SKY, skyFallback);

        bool reflectionProbesFallback = ssgiVolume.IsFallbackReflectionProbes();
        SetKeyword(_RAYMARCHING_FALLBACK_REFLECTION_PROBES, reflectionProbesFallback);

#if UNITY_6000_1_OR_NEWER
        bool hasProbeAtlas = Shader.IsKeywordEnabled(_CLUSTER_LIGHT_LOOP) && Shader.IsKeywordEnabled(_REFLECTION_PROBE_ATLAS);
#else
        bool hasProbeAtlas = Shader.IsKeywordEnabled(_FORWARD_PLUS);
#endif
        SetKeyword(_FP_REFL_PROBE_ATLAS, hasProbeAtlas && reflectionProbesFallback);
        m_SSGIPass.hasProbeAtlas = hasProbeAtlas;

        // APV settings
#if UNITY_2023_1_OR_NEWER
        bool outputAPVLighting = m_OverrideAmbientLighting && skyFallback && 
                                  (Shader.IsKeywordEnabled(PROBE_VOLUMES_L1) || Shader.IsKeywordEnabled(PROBE_VOLUMES_L2));
        SetKeyword(_APV_LIGHTING_BUFFER, outputAPVLighting);
        m_SSGIPass.outputAPVLighting = outputAPVLighting;
#else
        m_SSGIPass.outputAPVLighting = false;
        DisableKeyword(_APV_LIGHTING_BUFFER);
#endif

        // Backface settings
        bool renderBackfaceData = ssgiVolume.thicknessMode.value != ScreenSpaceGlobalIlluminationVolume.ThicknessMode.Constant;
        SetKeyword(_BACKFACE_TEXTURES, renderBackfaceData);
        
        bool isUsingDeferred = gBufferFieldInfo.GetValue(renderingData.cameraData.renderer) != null;
        isUsingDeferred &= (SystemInfo.graphicsDeviceType != GraphicsDeviceType.OpenGLES3) && 
                           (SystemInfo.graphicsDeviceType != GraphicsDeviceType.OpenGLCore);
        
        bool supportBackfaceLighting = m_BackfaceLighting && !isUsingDeferred && renderBackfaceData;
        m_ComputeShader.SetFloat(_BackDepthEnabled, renderBackfaceData ? (supportBackfaceLighting ? 2.0f : 1.0f) : 0.0f);

        // Camera settings
        bool isReflectionProbe = renderingData.cameraData.camera.cameraType == CameraType.Reflection;
        m_ComputeShader.SetFloat(_IsProbeCamera, isReflectionProbe ? 1.0f : 0.0f);

        // Upscaling
        SetKeyword(_DEPTH_NORMALS_UPSCALE, m_HighQualityUpscaling);
    }

    private void SetKeyword(string keyword, bool enabled)
    {
        if (enabled)
            m_ComputeShader.EnableKeyword(keyword);
        else
            m_ComputeShader.DisableKeyword(keyword);
    }

    private void EnableKeyword(string keyword)
    {
        m_ComputeShader.EnableKeyword(keyword);
    }

    private void DisableKeyword(string keyword)
    {
        m_ComputeShader.DisableKeyword(keyword);
    }

    #region Render Passes

    /// <summary>
    /// Pre-render pass for scene view motion vector fix (Editor only)
    /// </summary>
    public class PreRenderScreenSpaceGlobalIlluminationPass : ScriptableRenderPass
    {
        private const string m_ProfilerTag = "Prepare Screen Space Global Illumination";
        private readonly ProfilingSampler m_ProfilingSampler = new ProfilingSampler(m_ProfilerTag);

        public ComputeShader computeShader;

        private Matrix4x4 camVPMatrix;
        private Matrix4x4 prevCamVPMatrix;

        public PreRenderScreenSpaceGlobalIlluminationPass() { }

#if UNITY_6000_0_OR_NEWER
        [Obsolete]
#endif
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get();
            using (new ProfilingScope(cmd, m_ProfilingSampler))
            {
                cmd.SetGlobalMatrix(_PrevViewProjMatrix, prevCamVPMatrix);
                cmd.SetGlobalMatrix(_NonJitteredViewProjMatrix, camVPMatrix);
                prevCamVPMatrix = camVPMatrix;
            }
            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);
        }

#if UNITY_6000_0_OR_NEWER
        [Obsolete]
#endif
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            var cameraData = renderingData.cameraData;
            var camera = cameraData.camera;
            camVPMatrix = GL.GetGPUProjectionMatrix(camera.nonJitteredProjectionMatrix, true) * cameraData.GetViewMatrix();
            prevCamVPMatrix = prevCamVPMatrix == null ? camera.previousViewProjectionMatrix : prevCamVPMatrix;
        }

#if UNITY_6000_0_OR_NEWER
        private class PassData
        {
            internal Matrix4x4 prevCamVPMatrix;
            internal Matrix4x4 camVPMatrix;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            CommandBuffer cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            cmd.SetGlobalMatrix(_PrevViewProjMatrix, data.prevCamVPMatrix);
            cmd.SetGlobalMatrix(_NonJitteredViewProjMatrix, data.camVPMatrix);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using (var builder = renderGraph.AddUnsafePass<PassData>(m_ProfilerTag, out var passData))
            {
                UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
                var camera = cameraData.camera;
                
                camVPMatrix = GL.GetGPUProjectionMatrix(camera.nonJitteredProjectionMatrix, true) * cameraData.GetViewMatrix();
                passData.camVPMatrix = camVPMatrix;
                passData.prevCamVPMatrix = prevCamVPMatrix == null ? camera.previousViewProjectionMatrix : prevCamVPMatrix;
                prevCamVPMatrix = camVPMatrix;

                builder.AllowGlobalStateModification(true);
                builder.AllowPassCulling(false);
                builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
            }
        }
#endif

        public void Dispose() { }
    }

    /// <summary>
    /// Main SSGI compute shader pass
    /// </summary>
    public class ScreenSpaceGlobalIlluminationPass : ScriptableRenderPass
    {
        private const string m_ProfilerTag = "Screen Space Global Illumination";
        private readonly ProfilingSampler m_ProfilingSampler = new ProfilingSampler(m_ProfilerTag);

        private const int THREAD_GROUP_SIZE_X = 8;
        private const int THREAD_GROUP_SIZE_Y = 8;

        public ComputeShader computeShader;
        public ScreenSpaceGlobalIlluminationVolume ssgiVolume;
        public bool enableRenderingLayers;
        public bool overrideAmbientLighting;
        public bool hasProbeAtlas;
        public bool outputAPVLighting;
        public bool highQualityUpscaling;

        // RTHandles
        private RTHandle m_IntermediateCameraColorHandle;
        private RTHandle m_CombinedOutputHandle;
        private RTHandle m_DiffuseHandle;
        private RTHandle m_IntermediateDiffuseHandle;
        private RTHandle m_AccumulateSampleHandle;
        private RTHandle m_APVLightingHandle;

        // Persistent history handles
        private RTHandle m_HistoryDepthHandle;
        private RTHandle m_HistoryCameraColorHandle;
        private RTHandle m_HistoryIndirectDiffuseHandle;
        private RTHandle m_AccumulateHistorySampleHandle;

        private bool isHistoryTextureValid;
        private bool enableDenoise;
        private int frameCount = 0;
        private float resolutionScale = 1.0f;

        // Blur rotator values
        public static readonly float[] k_BlurRands = new float[] { 
            0.61264f, 0.296032f, 0.637552f, 0.524287f, 0.493583f, 0.972775f, 0.292517f, 0.771358f, 
            0.526745f, 0.769914f, 0.400229f, 0.891529f, 0.283315f, 0.352458f, 0.807725f, 0.919026f, 
            0.0697553f, 0.949327f, 0.525995f, 0.0860558f, 0.192214f, 0.663227f, 0.890233f, 0.348893f, 
            0.0641713f, 0.020023f, 0.457702f, 0.0630958f, 0.23828f, 0.970634f, 0.902208f, 0.85092f 
        };

        // Per-camera history data
        private const int MAX_CAMERA_COUNT = 4;
        private readonly CameraHistoryData[] cameraHistoryData = new CameraHistoryData[MAX_CAMERA_COUNT];
        private int cameraHistoryIndex;

        private struct CameraHistoryData
        {
            public int hash;
            public Matrix4x4 prevCamInvVPMatrix;
            public Vector3 prevCameraPositionWS;
            public float scaledWidth;
            public float scaledHeight;
            public RTHandle historyDepthHandle;
            public RTHandle historyCameraColorHandle;
            public RTHandle historyIndirectDiffuseHandle;
            public RTHandle accumulateHistorySampleHandle;
        }

        public ScreenSpaceGlobalIlluminationPass() { }

        #region Non Render Graph Pass
#if UNITY_6000_0_OR_NEWER
        [Obsolete]
#endif
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            SetupPass(renderingData.cameraData, renderingData.cullResults);
            ConfigureInput(ScriptableRenderPassInput.Motion);
        }

#if UNITY_6000_0_OR_NEWER
        [Obsolete]
#endif
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            RTHandle colorHandle = renderingData.cameraData.renderer.cameraColorTargetHandle;
            ExecuteCompute(context, renderingData.cameraData, colorHandle);
        }

        private void SetupPass(CameraData cameraData, CullingResults cullResults)
        {
            var camera = cameraData.camera;
            int currentCameraHash = camera.GetHashCode();
            cameraHistoryIndex = GetCameraHistoryDataIndex(currentCameraHash);

            if (!hasProbeAtlas)
            {
                var visibleReflectionProbes = cullResults.visibleReflectionProbes;
                UpdateReflectionProbe(visibleReflectionProbes, camera.transform.position);
            }
            else
            {
                computeShader.SetFloat(_ProbeSet, 0.0f);
            }

            computeShader.SetFloat(_FrameIndex, frameCount);
            computeShader.SetVector(_ReBlurBlurRotator, EvaluateRotator(k_BlurRands[frameCount % 32]));
            frameCount = (frameCount + 33) % 64000;

            int width = (int)(camera.scaledPixelWidth * cameraData.renderScale);
            int height = (int)(camera.scaledPixelHeight * cameraData.renderScale);

            bool denoiseStateChanged = ssgiVolume.denoiseSS.value != enableDenoise;
            bool resolutionStateChanged = ssgiVolume.fullResolutionSS.value ? resolutionScale != 1.0f : ssgiVolume.resolutionScaleSS.value != resolutionScale;
            bool cameraHasChanged = cameraHistoryIndex == -1;

            UpdateCameraHistoryData(cameraHasChanged);
            cameraHistoryIndex = cameraHasChanged ? 0 : cameraHistoryIndex;

            ref var historyData = ref cameraHistoryData[cameraHistoryIndex];

            if (historyData.prevCamInvVPMatrix != default)
                computeShader.SetMatrix(_PrevInvViewProjMatrix, historyData.prevCamInvVPMatrix);
            else
                computeShader.SetMatrix(_PrevInvViewProjMatrix, camera.previousViewProjectionMatrix.inverse);

            if (historyData.prevCameraPositionWS != default)
                computeShader.SetVector(_PrevCameraPositionWS, historyData.prevCameraPositionWS);
            else
                computeShader.SetVector(_PrevCameraPositionWS, camera.transform.position);

            historyData.prevCamInvVPMatrix = (GL.GetGPUProjectionMatrix(camera.projectionMatrix, true) * cameraData.GetViewMatrix()).inverse;
            historyData.prevCameraPositionWS = camera.transform.position;
            historyData.hash = currentCameraHash;

            float fieldOfView = camera.orthographic ? 1.0f : camera.fieldOfView;
            computeShader.SetFloat(_PixelSpreadAngleTangent, 
                Mathf.Tan(fieldOfView * Mathf.Deg2Rad * 0.5f) * 2.0f / 
                Mathf.Min(Mathf.FloorToInt(camera.scaledPixelWidth * resolutionScale), 
                          Mathf.FloorToInt(camera.scaledPixelHeight * resolutionScale)));

            resolutionStateChanged |= (historyData.scaledWidth != width) || (historyData.scaledHeight != height);
            if (!cameraHasChanged && (denoiseStateChanged || resolutionStateChanged))
                isHistoryTextureValid = false;

            historyData.scaledWidth = width;
            historyData.scaledHeight = height;

            resolutionScale = ssgiVolume.fullResolutionSS.value ? 1.0f : ssgiVolume.resolutionScaleSS.value;
            computeShader.SetFloat(_DownSample, resolutionScale);

            enableDenoise = ssgiVolume.denoiseSS.value;

            // Set ambient SH
            if (overrideAmbientLighting)
            {
                SphericalHarmonicsL2 ambientProbe = RenderSettings.ambientProbe;
                computeShader.SetVector(ssgi_SHAr, new Vector4(ambientProbe[0, 3], ambientProbe[0, 1], ambientProbe[0, 2], ambientProbe[0, 0] - ambientProbe[0, 6]));
                computeShader.SetVector(ssgi_SHAg, new Vector4(ambientProbe[1, 3], ambientProbe[1, 1], ambientProbe[1, 2], ambientProbe[1, 0] - ambientProbe[1, 6]));
                computeShader.SetVector(ssgi_SHAb, new Vector4(ambientProbe[2, 3], ambientProbe[2, 1], ambientProbe[2, 2], ambientProbe[2, 0] - ambientProbe[2, 6]));
                computeShader.SetVector(ssgi_SHBr, new Vector4(ambientProbe[0, 4], ambientProbe[0, 5], ambientProbe[0, 6] * 3, ambientProbe[0, 7]));
                computeShader.SetVector(ssgi_SHBg, new Vector4(ambientProbe[1, 4], ambientProbe[1, 5], ambientProbe[1, 6] * 3, ambientProbe[1, 7]));
                computeShader.SetVector(ssgi_SHBb, new Vector4(ambientProbe[2, 4], ambientProbe[2, 5], ambientProbe[2, 6] * 3, ambientProbe[2, 7]));
                computeShader.SetVector(ssgi_SHC, new Vector4(ambientProbe[0, 8], ambientProbe[1, 8], ambientProbe[2, 8], 1));
            }

            // Allocate render textures
            AllocateRenderTextures(width, height, ref historyData);
        }

        private void AllocateRenderTextures(int width, int height, ref CameraHistoryData historyData)
        {
            RenderTextureDescriptor desc = new RenderTextureDescriptor(width, height);
            desc.graphicsFormat = GraphicsFormat.B10G11R11_UFloatPack32;
            desc.depthBufferBits = 0;
            desc.stencilFormat = GraphicsFormat.None;
            desc.depthStencilFormat = GraphicsFormat.None;
            desc.msaaSamples = 1;
            desc.bindMS = false;
            desc.enableRandomWrite = true;

            RenderTextureDescriptor depthDesc = desc;

#if UNITY_6000_0_OR_NEWER
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_IntermediateCameraColorHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_IntermediateCameraColorTexture");
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_CombinedOutputHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_CombinedOutputTexture");
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_APVLightingHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_APVLightingTexture");
#else
            RenderingUtils.ReAllocateIfNeeded(ref m_IntermediateCameraColorHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_IntermediateCameraColorTexture");
            RenderingUtils.ReAllocateIfNeeded(ref m_CombinedOutputHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_CombinedOutputTexture");
            RenderingUtils.ReAllocateIfNeeded(ref m_APVLightingHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_APVLightingTexture");
#endif

            int ssgiWidth = Mathf.FloorToInt(width * resolutionScale);
            int ssgiHeight = Mathf.FloorToInt(height * resolutionScale);
            desc.width = ssgiWidth;
            desc.height = ssgiHeight;

#if UNITY_6000_0_OR_NEWER
            RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.historyCameraColorHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistoryCameraColorTexture");
#else
            RenderingUtils.ReAllocateIfNeeded(ref historyData.historyCameraColorHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistoryCameraColorTexture");
#endif

            if (isHistoryTextureValid)
            {
                computeShader.SetFloat(_HistoryTextureValid, 1.0f);
                computeShader.SetTexture(KernelIndices.SSGIRayMarching, _SSGIHistoryCameraColorTexture, historyData.historyCameraColorHandle);
            }
            else
            {
                computeShader.SetFloat(_HistoryTextureValid, 0.0f);
                isHistoryTextureValid = true;
            }

            desc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;
            depthDesc.width = ssgiWidth;
            depthDesc.height = ssgiHeight;
            depthDesc.graphicsFormat = GraphicsFormat.R32_SFloat;
            depthDesc.enableRandomWrite = true;

#if UNITY_6000_0_OR_NEWER
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_DiffuseHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_IndirectDiffuseTexture");
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_IntermediateDiffuseHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_IntermediateIndirectDiffuseTexture");
            RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.historyIndirectDiffuseHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_HistoryIndirectDiffuseTexture");
            RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.historyDepthHandle, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistoryDepthTexture");
#else
            RenderingUtils.ReAllocateIfNeeded(ref m_DiffuseHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_IndirectDiffuseTexture");
            RenderingUtils.ReAllocateIfNeeded(ref m_IntermediateDiffuseHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_IntermediateIndirectDiffuseTexture");
            RenderingUtils.ReAllocateIfNeeded(ref historyData.historyIndirectDiffuseHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_HistoryIndirectDiffuseTexture");
            RenderingUtils.ReAllocateIfNeeded(ref historyData.historyDepthHandle, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistoryDepthTexture");
#endif

            desc.graphicsFormat = GraphicsFormat.R16_SFloat;

#if UNITY_6000_0_OR_NEWER
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_AccumulateSampleHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGISampleTexture");
            RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.accumulateHistorySampleHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistorySampleTexture");
#else
            RenderingUtils.ReAllocateIfNeeded(ref m_AccumulateSampleHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGISampleTexture");
            RenderingUtils.ReAllocateIfNeeded(ref historyData.accumulateHistorySampleHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistorySampleTexture");
#endif

            // Set texel size
            computeShader.SetVector(_IndirectDiffuseTexture_TexelSize, 
                new Vector4(1.0f / ssgiWidth, 1.0f / ssgiHeight, ssgiWidth, ssgiHeight));
        }

        private void ExecuteCompute(ScriptableRenderContext context, CameraData cameraData, RTHandle colorHandle)
        {
            var camera = cameraData.camera;
            int width = (int)(camera.scaledPixelWidth * cameraData.renderScale);
            int height = (int)(camera.scaledPixelHeight * cameraData.renderScale);
            int ssgiWidth = Mathf.FloorToInt(width * resolutionScale);
            int ssgiHeight = Mathf.FloorToInt(height * resolutionScale);

            ref var historyData = ref cameraHistoryData[cameraHistoryIndex];

            CommandBuffer cmd = CommandBufferPool.Get();
            using (new ProfilingScope(cmd, m_ProfilingSampler))
            {
                // Set screen size
                cmd.SetComputeVectorParam(computeShader, _ScreenSize, new Vector4(width, height, 1.0f / width, 1.0f / height));
                cmd.SetComputeVectorParam(computeShader, _SSGITextureSize, new Vector4(ssgiWidth, ssgiHeight, 1.0f / ssgiWidth, 1.0f / ssgiHeight));

                int threadGroupsX = Mathf.CeilToInt(width / (float)THREAD_GROUP_SIZE_X);
                int threadGroupsY = Mathf.CeilToInt(height / (float)THREAD_GROUP_SIZE_Y);
                int ssgiThreadGroupsX = Mathf.CeilToInt(ssgiWidth / (float)THREAD_GROUP_SIZE_X);
                int ssgiThreadGroupsY = Mathf.CeilToInt(ssgiHeight / (float)THREAD_GROUP_SIZE_Y);
                RTHandle cameraDepthHandle = cameraData.renderer.cameraDepthTargetHandle;
                Texture motionVectorRT = Shader.GetGlobalTexture("_MotionVectorTexture");

                cmd.SetComputeTextureParam(computeShader, KernelIndices.CopyDirectLighting, _CameraDepthTexture, cameraDepthHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.SSGIRayMarching, _CameraDepthTexture, cameraDepthHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalReprojection, _CameraDepthTexture, cameraDepthHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.SpatialDenoise, _CameraDepthTexture, cameraDepthHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalStabilization, _CameraDepthTexture, cameraDepthHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.CopyHistoryDepth, _CameraDepthTexture, cameraDepthHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.CombineGI, _CameraDepthTexture, cameraDepthHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.PoissonDiskDenoise, _CameraDepthTexture, cameraDepthHandle);

                cmd.SetComputeTextureParam(computeShader, KernelIndices.SSGIRayMarching, _MotionVectorTexture, motionVectorRT);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalReprojection, _MotionVectorTexture, motionVectorRT);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalStabilization, _MotionVectorTexture, motionVectorRT);

                // Kernel 0: Copy Direct Lighting
                cmd.SetComputeTextureParam(computeShader, KernelIndices.CopyDirectLighting, _SourceTexture, colorHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.CopyDirectLighting, _RW_IntermediateCameraColorTexture, m_IntermediateCameraColorHandle);
                if (outputAPVLighting)
                    cmd.SetComputeTextureParam(computeShader, KernelIndices.CopyDirectLighting, _RW_APVLightingTexture, m_APVLightingHandle);
                cmd.DispatchCompute(computeShader, KernelIndices.CopyDirectLighting, threadGroupsX, threadGroupsY, 1);

                // Kernel 1: SSGI Ray Marching
                cmd.SetComputeTextureParam(computeShader, KernelIndices.SSGIRayMarching, _IntermediateCameraColorTexture, m_IntermediateCameraColorHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.SSGIRayMarching, _SSGIHistoryCameraColorTexture, historyData.historyCameraColorHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.SSGIRayMarching, _SSGIHistoryDepthTexture, historyData.historyDepthHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.SSGIRayMarching, _RW_IntermediateIndirectDiffuseTexture, m_IntermediateDiffuseHandle);
                cmd.DispatchCompute(computeShader, KernelIndices.SSGIRayMarching, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);

                if (enableDenoise)
                {
                    // Kernel 2: Temporal Reprojection
                    cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalReprojection, _IntermediateIndirectDiffuseTexture, m_IntermediateDiffuseHandle);
                    cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalReprojection, _HistoryIndirectDiffuseTexture, historyData.historyIndirectDiffuseHandle);
                    cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalReprojection, _SSGIHistorySampleTexture, historyData.accumulateHistorySampleHandle);
                    cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalReprojection, _RW_IndirectDiffuseTexture, m_DiffuseHandle);
                    cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalReprojection, _RW_SSGISampleTexture, m_AccumulateSampleHandle);
                    cmd.DispatchCompute(computeShader, KernelIndices.TemporalReprojection, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);

                    // Aggressive denoise (optional)
                    if (ssgiVolume.denoiserAlgorithmSS.value == ScreenSpaceGlobalIlluminationVolume.DenoiserAlgorithm.Aggressive)
                    {
                        // First pass
                        cmd.SetComputeTextureParam(computeShader, KernelIndices.PoissonDiskDenoise, _SourceDiffuseTexture, m_DiffuseHandle);
                        cmd.SetComputeTextureParam(computeShader, KernelIndices.PoissonDiskDenoise, _RW_IntermediateIndirectDiffuseTexture, m_IntermediateDiffuseHandle);
                        cmd.DispatchCompute(computeShader, KernelIndices.PoissonDiskDenoise, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);

                        // Second pass
                        cmd.SetComputeTextureParam(computeShader, KernelIndices.PoissonDiskDenoise, _SourceDiffuseTexture, m_IntermediateDiffuseHandle);
                        cmd.SetComputeTextureParam(computeShader, KernelIndices.PoissonDiskDenoise, _RW_IntermediateIndirectDiffuseTexture, m_DiffuseHandle);
                        cmd.DispatchCompute(computeShader, KernelIndices.PoissonDiskDenoise, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);
                    }

                    // Second denoiser pass (optional)
                    if (ssgiVolume.secondDenoiserPassSS.value)
                    {
                        // Kernel 3: Spatial Denoise
                        cmd.SetComputeTextureParam(computeShader, KernelIndices.SpatialDenoise, _SourceDiffuseTexture, m_DiffuseHandle);
                        cmd.SetComputeTextureParam(computeShader, KernelIndices.SpatialDenoise, _RW_IntermediateIndirectDiffuseTexture, m_IntermediateDiffuseHandle);
                        cmd.DispatchCompute(computeShader, KernelIndices.SpatialDenoise, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);

                        // Kernel 4: Temporal Stabilization
                        cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalStabilization, _SourceDiffuseTexture, m_IntermediateDiffuseHandle);
                        cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalStabilization, _HistoryIndirectDiffuseTexture, historyData.historyIndirectDiffuseHandle);
                        cmd.SetComputeTextureParam(computeShader, KernelIndices.TemporalStabilization, _RW_IndirectDiffuseTexture, m_DiffuseHandle);
                        cmd.DispatchCompute(computeShader, KernelIndices.TemporalStabilization, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);
                    }

                    // Copy to history
                    cmd.CopyTexture(m_DiffuseHandle, historyData.historyIndirectDiffuseHandle);
                    cmd.CopyTexture(m_AccumulateSampleHandle, historyData.accumulateHistorySampleHandle);
                }
                else
                {
                    // No denoise - copy ray marching result directly
                    cmd.CopyTexture(m_IntermediateDiffuseHandle, m_DiffuseHandle);
                }

                // Kernel 5: Copy History Depth
                cmd.SetComputeTextureParam(computeShader, KernelIndices.CopyHistoryDepth, _RW_SSGIHistoryDepthTexture, historyData.historyDepthHandle);
                cmd.DispatchCompute(computeShader, KernelIndices.CopyHistoryDepth, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);

                // Kernel 6: Combine GI
                cmd.SetComputeTextureParam(computeShader, KernelIndices.CombineGI, _IndirectDiffuseTexture, m_DiffuseHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.CombineGI, _IntermediateCameraColorTexture, m_IntermediateCameraColorHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.CombineGI, _RW_CameraColorTexture, m_CombinedOutputHandle);
                cmd.DispatchCompute(computeShader, KernelIndices.CombineGI, threadGroupsX, threadGroupsY, 1);
                cmd.Blit(m_CombinedOutputHandle, colorHandle);

                // Kernel 9: Copy to history camera color
                cmd.SetComputeTextureParam(computeShader, KernelIndices.BlitColorTexture, _SourceTexture, colorHandle);
                cmd.SetComputeTextureParam(computeShader, KernelIndices.BlitColorTexture, _RW_DestinationTexture, historyData.historyCameraColorHandle);
                cmd.DispatchCompute(computeShader, KernelIndices.BlitColorTexture, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);
            }

            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);
        }
        #endregion

        #region Render Graph Pass
#if UNITY_6000_0_OR_NEWER
        private class PassData
        {
            internal ComputeShader computeShader;
            internal ScreenSpaceGlobalIlluminationVolume ssgiVolume;
            internal bool enableDenoise;
            internal bool outputAPVLighting;
            internal bool overrideAmbientLighting;

            internal int width;
            internal int height;
            internal int ssgiWidth;
            internal int ssgiHeight;

            internal TextureHandle cameraColorHandle;
            internal TextureHandle cameraDepthHandle;
            internal TextureHandle motionVectorHandle;

            internal TextureHandle intermediateCameraColorHandle;
            internal TextureHandle combinedOutputHandle;
            internal TextureHandle diffuseHandle;
            internal TextureHandle intermediateDiffuseHandle;
            internal TextureHandle accumulateSampleHandle;
            internal TextureHandle apvLightingHandle;

            internal TextureHandle historyDepthHandle;
            internal TextureHandle historyCameraColorHandle;
            internal TextureHandle historyIndirectDiffuseHandle;
            internal TextureHandle accumulateHistorySampleHandle;

            internal TextureHandle gBuffer0Handle;
            internal TextureHandle gBuffer1Handle;
            internal TextureHandle gBuffer2Handle;
            internal bool localGBuffers;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            CommandBuffer cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var cs = data.computeShader;

            int threadGroupsX = Mathf.CeilToInt(data.width / (float)THREAD_GROUP_SIZE_X);
            int threadGroupsY = Mathf.CeilToInt(data.height / (float)THREAD_GROUP_SIZE_Y);
            int ssgiThreadGroupsX = Mathf.CeilToInt(data.ssgiWidth / (float)THREAD_GROUP_SIZE_X);
            int ssgiThreadGroupsY = Mathf.CeilToInt(data.ssgiHeight / (float)THREAD_GROUP_SIZE_Y);

            cmd.SetComputeVectorParam(cs, _ScreenSize, new Vector4(data.width, data.height, 1.0f / data.width, 1.0f / data.height));
            cmd.SetComputeVectorParam(cs, _SSGITextureSize, new Vector4(data.ssgiWidth, data.ssgiHeight, 1.0f / data.ssgiWidth, 1.0f / data.ssgiHeight));

            // Set GBuffer textures
            if (data.localGBuffers)
            {
                cmd.SetComputeTextureParam(cs, KernelIndices.CopyDirectLighting, _GBuffer0, data.gBuffer0Handle);
                cmd.SetComputeTextureParam(cs, KernelIndices.CopyDirectLighting, _GBuffer1, data.gBuffer1Handle);
                cmd.SetComputeTextureParam(cs, KernelIndices.CopyDirectLighting, _GBuffer2, data.gBuffer2Handle);
                cmd.SetComputeTextureParam(cs, KernelIndices.SSGIRayMarching, _GBuffer2, data.gBuffer2Handle);
                cmd.SetComputeTextureParam(cs, KernelIndices.SpatialDenoise, _GBuffer2, data.gBuffer2Handle);
                cmd.SetComputeTextureParam(cs, KernelIndices.CombineGI, _GBuffer0, data.gBuffer0Handle);
                cmd.SetComputeTextureParam(cs, KernelIndices.CombineGI, _GBuffer1, data.gBuffer1Handle);
                cmd.SetComputeTextureParam(cs, KernelIndices.CombineGI, _GBuffer2, data.gBuffer2Handle);
                cmd.SetComputeTextureParam(cs, KernelIndices.PoissonDiskDenoise, _GBuffer2, data.gBuffer2Handle);
            }

            // Set common textures
            cmd.SetComputeTextureParam(cs, KernelIndices.CopyDirectLighting, _CameraDepthTexture, data.cameraDepthHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.SSGIRayMarching, _CameraDepthTexture, data.cameraDepthHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.SSGIRayMarching, _MotionVectorTexture, data.motionVectorHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.TemporalReprojection, _CameraDepthTexture, data.cameraDepthHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.TemporalReprojection, _MotionVectorTexture, data.motionVectorHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.SpatialDenoise, _CameraDepthTexture, data.cameraDepthHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.TemporalStabilization, _CameraDepthTexture, data.cameraDepthHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.TemporalStabilization, _MotionVectorTexture, data.motionVectorHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.CopyHistoryDepth, _CameraDepthTexture, data.cameraDepthHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.CombineGI, _CameraDepthTexture, data.cameraDepthHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.PoissonDiskDenoise, _CameraDepthTexture, data.cameraDepthHandle);

            // Kernel 0: Copy Direct Lighting
            cmd.SetComputeTextureParam(cs, KernelIndices.CopyDirectLighting, _SourceTexture, data.cameraColorHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.CopyDirectLighting, _RW_IntermediateCameraColorTexture, data.intermediateCameraColorHandle);
            if (data.outputAPVLighting)
                cmd.SetComputeTextureParam(cs, KernelIndices.CopyDirectLighting, _RW_APVLightingTexture, data.apvLightingHandle);
            cmd.DispatchCompute(cs, KernelIndices.CopyDirectLighting, threadGroupsX, threadGroupsY, 1);

            // Kernel 1: SSGI Ray Marching
            cmd.SetComputeTextureParam(cs, KernelIndices.SSGIRayMarching, _IntermediateCameraColorTexture, data.intermediateCameraColorHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.SSGIRayMarching, _SSGIHistoryCameraColorTexture, data.historyCameraColorHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.SSGIRayMarching, _SSGIHistoryDepthTexture, data.historyDepthHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.SSGIRayMarching, _RW_IntermediateIndirectDiffuseTexture, data.intermediateDiffuseHandle);
            cmd.DispatchCompute(cs, KernelIndices.SSGIRayMarching, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);

            if (data.enableDenoise)
            {
                // Kernel 2: Temporal Reprojection
                cmd.SetComputeTextureParam(cs, KernelIndices.TemporalReprojection, _IntermediateIndirectDiffuseTexture, data.intermediateDiffuseHandle);
                cmd.SetComputeTextureParam(cs, KernelIndices.TemporalReprojection, _HistoryIndirectDiffuseTexture, data.historyIndirectDiffuseHandle);
                cmd.SetComputeTextureParam(cs, KernelIndices.TemporalReprojection, _SSGIHistorySampleTexture, data.accumulateHistorySampleHandle);
                cmd.SetComputeTextureParam(cs, KernelIndices.TemporalReprojection, _RW_IndirectDiffuseTexture, data.diffuseHandle);
                cmd.SetComputeTextureParam(cs, KernelIndices.TemporalReprojection, _RW_SSGISampleTexture, data.accumulateSampleHandle);
                cmd.DispatchCompute(cs, KernelIndices.TemporalReprojection, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);

                // Aggressive denoise
                if (data.ssgiVolume.denoiserAlgorithmSS.value == ScreenSpaceGlobalIlluminationVolume.DenoiserAlgorithm.Aggressive)
                {
                    cmd.SetComputeTextureParam(cs, KernelIndices.PoissonDiskDenoise, _SourceDiffuseTexture, data.diffuseHandle);
                    cmd.SetComputeTextureParam(cs, KernelIndices.PoissonDiskDenoise, _RW_IntermediateIndirectDiffuseTexture, data.intermediateDiffuseHandle);
                    cmd.DispatchCompute(cs, KernelIndices.PoissonDiskDenoise, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);

                    cmd.SetComputeTextureParam(cs, KernelIndices.PoissonDiskDenoise, _SourceDiffuseTexture, data.intermediateDiffuseHandle);
                    cmd.SetComputeTextureParam(cs, KernelIndices.PoissonDiskDenoise, _RW_IntermediateIndirectDiffuseTexture, data.diffuseHandle);
                    cmd.DispatchCompute(cs, KernelIndices.PoissonDiskDenoise, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);
                }

                // Second denoiser pass
                if (data.ssgiVolume.secondDenoiserPassSS.value)
                {
                    cmd.SetComputeTextureParam(cs, KernelIndices.SpatialDenoise, _SourceDiffuseTexture, data.diffuseHandle);
                    cmd.SetComputeTextureParam(cs, KernelIndices.SpatialDenoise, _RW_IntermediateIndirectDiffuseTexture, data.intermediateDiffuseHandle);
                    cmd.DispatchCompute(cs, KernelIndices.SpatialDenoise, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);

                    cmd.SetComputeTextureParam(cs, KernelIndices.TemporalStabilization, _SourceDiffuseTexture, data.intermediateDiffuseHandle);
                    cmd.SetComputeTextureParam(cs, KernelIndices.TemporalStabilization, _HistoryIndirectDiffuseTexture, data.historyIndirectDiffuseHandle);
                    cmd.SetComputeTextureParam(cs, KernelIndices.TemporalStabilization, _RW_IndirectDiffuseTexture, data.diffuseHandle);
                    cmd.DispatchCompute(cs, KernelIndices.TemporalStabilization, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);
                }

                cmd.CopyTexture(data.diffuseHandle, data.historyIndirectDiffuseHandle);
                cmd.CopyTexture(data.accumulateSampleHandle, data.accumulateHistorySampleHandle);
            }
            else
            {
                cmd.CopyTexture(data.intermediateDiffuseHandle, data.diffuseHandle);
            }

            // Kernel 5: Copy History Depth
            cmd.SetComputeTextureParam(cs, KernelIndices.CopyHistoryDepth, _RW_SSGIHistoryDepthTexture, data.historyDepthHandle);
            cmd.DispatchCompute(cs, KernelIndices.CopyHistoryDepth, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);

            // Kernel 6: Combine GI
            cmd.SetComputeTextureParam(cs, KernelIndices.CombineGI, _IndirectDiffuseTexture, data.diffuseHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.CombineGI, _IntermediateCameraColorTexture, data.intermediateCameraColorHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.CombineGI, _RW_CameraColorTexture, data.combinedOutputHandle);
            cmd.DispatchCompute(cs, KernelIndices.CombineGI, threadGroupsX, threadGroupsY, 1);
            cmd.Blit(data.combinedOutputHandle, data.cameraColorHandle);

            // Copy to history
            cmd.SetComputeTextureParam(cs, KernelIndices.BlitColorTexture, _SourceTexture, data.cameraColorHandle);
            cmd.SetComputeTextureParam(cs, KernelIndices.BlitColorTexture, _RW_DestinationTexture, data.historyCameraColorHandle);
            cmd.DispatchCompute(cs, KernelIndices.BlitColorTexture, ssgiThreadGroupsX, ssgiThreadGroupsY, 1);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using (var builder = renderGraph.AddUnsafePass<PassData>(m_ProfilerTag, out var passData))
            {
                UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
                UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
                UniversalRenderingData renderingData = frameData.Get<UniversalRenderingData>();

                var camera = cameraData.camera;
                int currentCameraHash = camera.GetHashCode();
                int historyIndex = GetCameraHistoryDataIndex(currentCameraHash);

                if (!hasProbeAtlas)
                {
                    var visibleReflectionProbes = renderingData.cullResults.visibleReflectionProbes;
                    UpdateReflectionProbe(visibleReflectionProbes, camera.transform.position);
                }
                else
                {
                    computeShader.SetFloat(_ProbeSet, 0.0f);
                }

                computeShader.SetFloat(_FrameIndex, frameCount);
                computeShader.SetVector(_ReBlurBlurRotator, EvaluateRotator(k_BlurRands[frameCount % 32]));
                frameCount = (frameCount + 33) % 64000;

                int width = (int)(camera.scaledPixelWidth * cameraData.renderScale);
                int height = (int)(camera.scaledPixelHeight * cameraData.renderScale);

                bool denoiseStateChanged = ssgiVolume.denoiseSS.value != enableDenoise;
                bool resolutionStateChanged = ssgiVolume.fullResolutionSS.value ? resolutionScale != 1.0f : ssgiVolume.resolutionScaleSS.value != resolutionScale;
                bool cameraHasChanged = historyIndex == -1;

                UpdateCameraHistoryData(cameraHasChanged);
                historyIndex = cameraHasChanged ? 0 : historyIndex;
                cameraHistoryIndex = historyIndex;

                ref var historyData = ref cameraHistoryData[historyIndex];

                if (historyData.prevCamInvVPMatrix != default)
                    computeShader.SetMatrix(_PrevInvViewProjMatrix, historyData.prevCamInvVPMatrix);
                else
                    computeShader.SetMatrix(_PrevInvViewProjMatrix, camera.previousViewProjectionMatrix.inverse);

                if (historyData.prevCameraPositionWS != default)
                    computeShader.SetVector(_PrevCameraPositionWS, historyData.prevCameraPositionWS);
                else
                    computeShader.SetVector(_PrevCameraPositionWS, camera.transform.position);

                historyData.prevCamInvVPMatrix = (GL.GetGPUProjectionMatrix(camera.projectionMatrix, true) * cameraData.GetViewMatrix()).inverse;
                historyData.prevCameraPositionWS = camera.transform.position;
                historyData.hash = currentCameraHash;

                float fieldOfView = camera.orthographic ? 1.0f : camera.fieldOfView;
                computeShader.SetFloat(_PixelSpreadAngleTangent,
                    Mathf.Tan(fieldOfView * Mathf.Deg2Rad * 0.5f) * 2.0f /
                    Mathf.Min(Mathf.FloorToInt(camera.scaledPixelWidth * resolutionScale),
                              Mathf.FloorToInt(camera.scaledPixelHeight * resolutionScale)));

                resolutionStateChanged |= (historyData.scaledWidth != width) || (historyData.scaledHeight != height);
                if (!cameraHasChanged && (denoiseStateChanged || resolutionStateChanged))
                    isHistoryTextureValid = false;

                historyData.scaledWidth = width;
                historyData.scaledHeight = height;

                resolutionScale = ssgiVolume.fullResolutionSS.value ? 1.0f : ssgiVolume.resolutionScaleSS.value;
                computeShader.SetFloat(_DownSample, resolutionScale);
                enableDenoise = ssgiVolume.denoiseSS.value;

                int ssgiWidth = Mathf.FloorToInt(width * resolutionScale);
                int ssgiHeight = Mathf.FloorToInt(height * resolutionScale);

                // Set ambient SH
                if (overrideAmbientLighting)
                {
                    SphericalHarmonicsL2 ambientProbe = RenderSettings.ambientProbe;
                    computeShader.SetVector(ssgi_SHAr, new Vector4(ambientProbe[0, 3], ambientProbe[0, 1], ambientProbe[0, 2], ambientProbe[0, 0] - ambientProbe[0, 6]));
                    computeShader.SetVector(ssgi_SHAg, new Vector4(ambientProbe[1, 3], ambientProbe[1, 1], ambientProbe[1, 2], ambientProbe[1, 0] - ambientProbe[1, 6]));
                    computeShader.SetVector(ssgi_SHAb, new Vector4(ambientProbe[2, 3], ambientProbe[2, 1], ambientProbe[2, 2], ambientProbe[2, 0] - ambientProbe[2, 6]));
                    computeShader.SetVector(ssgi_SHBr, new Vector4(ambientProbe[0, 4], ambientProbe[0, 5], ambientProbe[0, 6] * 3, ambientProbe[0, 7]));
                    computeShader.SetVector(ssgi_SHBg, new Vector4(ambientProbe[1, 4], ambientProbe[1, 5], ambientProbe[1, 6] * 3, ambientProbe[1, 7]));
                    computeShader.SetVector(ssgi_SHBb, new Vector4(ambientProbe[2, 4], ambientProbe[2, 5], ambientProbe[2, 6] * 3, ambientProbe[2, 7]));
                    computeShader.SetVector(ssgi_SHC, new Vector4(ambientProbe[0, 8], ambientProbe[1, 8], ambientProbe[2, 8], 1));
                }

                // Create render graph textures
                RenderTextureDescriptor desc = cameraData.cameraTargetDescriptor;
                desc.graphicsFormat = GraphicsFormat.B10G11R11_UFloatPack32;
                desc.depthBufferBits = 0;
                desc.stencilFormat = GraphicsFormat.None;
                desc.msaaSamples = 1;
                desc.bindMS = false;
                desc.enableRandomWrite = true;

                TextureHandle intermediateCameraColorHandle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, desc, "_IntermediateCameraColorTexture", false, FilterMode.Point, TextureWrapMode.Clamp);
                TextureHandle combinedOutputHandle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, desc, "_CombinedOutputTexture", false, FilterMode.Point, TextureWrapMode.Clamp);
                TextureHandle apvLightingHandle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, desc, "_APVLightingTexture", false, FilterMode.Point, TextureWrapMode.Clamp);

                desc.width = ssgiWidth;
                desc.height = ssgiHeight;
                desc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;

                TextureHandle diffuseHandle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, desc, "_IndirectDiffuseTexture", false, FilterMode.Point, TextureWrapMode.Clamp);
                TextureHandle intermediateDiffuseHandle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, desc, "_IntermediateIndirectDiffuseTexture", false, FilterMode.Point, TextureWrapMode.Clamp);

                // Allocate persistent history textures
                RenderTextureDescriptor historyColorDesc = desc;
                historyColorDesc.graphicsFormat = GraphicsFormat.B10G11R11_UFloatPack32;
                RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.historyCameraColorHandle, historyColorDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistoryCameraColorTexture");

                historyColorDesc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;
                RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.historyIndirectDiffuseHandle, historyColorDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_HistoryIndirectDiffuseTexture");

                RenderTextureDescriptor depthDesc = desc;
                depthDesc.graphicsFormat = GraphicsFormat.R32_SFloat;
                RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.historyDepthHandle, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistoryDepthTexture");

                depthDesc.graphicsFormat = GraphicsFormat.R16_SFloat;
                RenderingUtils.ReAllocateHandleIfNeeded(ref m_AccumulateSampleHandle, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGISampleTexture");
                RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.accumulateHistorySampleHandle, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistorySampleTexture");

                if (isHistoryTextureValid)
                {
                    computeShader.SetFloat(_HistoryTextureValid, 1.0f);
                }
                else
                {
                    computeShader.SetFloat(_HistoryTextureValid, 0.0f);
                    isHistoryTextureValid = true;
                }

                computeShader.SetVector(_IndirectDiffuseTexture_TexelSize,
                    new Vector4(1.0f / ssgiWidth, 1.0f / ssgiHeight, ssgiWidth, ssgiHeight));

                // Import persistent textures
                TextureHandle historyCameraColorHandle = renderGraph.ImportTexture(historyData.historyCameraColorHandle);
                TextureHandle historyIndirectDiffuseHandle = renderGraph.ImportTexture(historyData.historyIndirectDiffuseHandle);
                TextureHandle historyDepthHandle = renderGraph.ImportTexture(historyData.historyDepthHandle);
                TextureHandle accumulateSampleHandle = renderGraph.ImportTexture(m_AccumulateSampleHandle);
                TextureHandle accumulateHistorySampleHandle = renderGraph.ImportTexture(historyData.accumulateHistorySampleHandle);

                // Fill pass data
                passData.computeShader = computeShader;
                passData.ssgiVolume = ssgiVolume;
                passData.enableDenoise = enableDenoise;
                passData.outputAPVLighting = outputAPVLighting;
                passData.overrideAmbientLighting = overrideAmbientLighting;
                passData.width = width;
                passData.height = height;
                passData.ssgiWidth = ssgiWidth;
                passData.ssgiHeight = ssgiHeight;

                passData.cameraColorHandle = resourceData.activeColorTexture;
                passData.cameraDepthHandle = resourceData.cameraDepthTexture;
                passData.motionVectorHandle = resourceData.motionVectorColor;
                passData.intermediateCameraColorHandle = intermediateCameraColorHandle;
                passData.combinedOutputHandle = combinedOutputHandle;
                passData.diffuseHandle = diffuseHandle;
                passData.intermediateDiffuseHandle = intermediateDiffuseHandle;
                passData.accumulateSampleHandle = accumulateSampleHandle;
                passData.apvLightingHandle = apvLightingHandle;
                passData.historyDepthHandle = historyDepthHandle;
                passData.historyCameraColorHandle = historyCameraColorHandle;
                passData.historyIndirectDiffuseHandle = historyIndirectDiffuseHandle;
                passData.accumulateHistorySampleHandle = accumulateHistorySampleHandle;

                passData.localGBuffers = resourceData.gBuffer[0].IsValid();
                if (passData.localGBuffers)
                {
                    passData.gBuffer0Handle = resourceData.gBuffer[0];
                    passData.gBuffer1Handle = resourceData.gBuffer[1];
                    passData.gBuffer2Handle = resourceData.gBuffer[2];
                    builder.UseTexture(passData.gBuffer0Handle, AccessFlags.Read);
                    builder.UseTexture(passData.gBuffer1Handle, AccessFlags.Read);
                    builder.UseTexture(passData.gBuffer2Handle, AccessFlags.Read);
                }

                // Declare texture usage
                builder.UseTexture(passData.cameraColorHandle, AccessFlags.ReadWrite);
                builder.UseTexture(passData.combinedOutputHandle, AccessFlags.ReadWrite);
                builder.UseTexture(passData.cameraDepthHandle, AccessFlags.Read);
                builder.UseTexture(passData.motionVectorHandle, AccessFlags.Read);
                builder.UseTexture(passData.intermediateCameraColorHandle, AccessFlags.ReadWrite);
                builder.UseTexture(passData.diffuseHandle, AccessFlags.ReadWrite);
                builder.UseTexture(passData.intermediateDiffuseHandle, AccessFlags.ReadWrite);
                builder.UseTexture(passData.accumulateSampleHandle, AccessFlags.ReadWrite);
                builder.UseTexture(passData.apvLightingHandle, AccessFlags.Write);
                builder.UseTexture(passData.historyDepthHandle, AccessFlags.ReadWrite);
                builder.UseTexture(passData.historyCameraColorHandle, AccessFlags.ReadWrite);
                builder.UseTexture(passData.historyIndirectDiffuseHandle, AccessFlags.ReadWrite);
                builder.UseTexture(passData.accumulateHistorySampleHandle, AccessFlags.ReadWrite);

                builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
            }
        }
#endif
        #endregion

        #region Helpers
        private int GetCameraHistoryDataIndex(int cameraHash)
        {
            for (int i = 0; i < MAX_CAMERA_COUNT; i++)
            {
                if (cameraHistoryData[i].hash == cameraHash)
                    return i;
            }
            return -1;
        }

        private void UpdateCameraHistoryData(bool cameraHashChanged)
        {
            if (cameraHashChanged)
            {
                const int lastIndex = MAX_CAMERA_COUNT - 1;

                cameraHistoryData[lastIndex].historyDepthHandle?.Release();
                cameraHistoryData[lastIndex].historyCameraColorHandle?.Release();
                cameraHistoryData[lastIndex].historyIndirectDiffuseHandle?.Release();
                cameraHistoryData[lastIndex].accumulateHistorySampleHandle?.Release();

                Array.Copy(cameraHistoryData, 0, cameraHistoryData, 1, lastIndex);
                cameraHistoryData[0] = default;
            }
        }

        private void UpdateReflectionProbe(NativeArray<VisibleReflectionProbe> visibleReflectionProbes, Vector3 cameraPosition)
        {
            var reflectionProbe = GetClosestProbe(visibleReflectionProbes, cameraPosition);
            if (reflectionProbe != null)
            {
                computeShader.SetTexture(KernelIndices.SSGIRayMarching, _SpecCube0, reflectionProbe.texture);
                computeShader.SetVector(_SpecCube0_HDR, reflectionProbe.textureHDRDecodeValues);

                bool isBoxProjected = reflectionProbe.boxProjection;
                if (isBoxProjected)
                {
                    Vector3 probe0Position = reflectionProbe.transform.position;
                    computeShader.SetVector(_SpecCube0_BoxMin, reflectionProbe.bounds.min);
                    computeShader.SetVector(_SpecCube0_BoxMax, reflectionProbe.bounds.max);
                    computeShader.SetVector(_SpecCube0_ProbePosition, new Vector4(probe0Position.x, probe0Position.y, probe0Position.z, 1.0f));
                }
                else
                {
                    computeShader.SetVector(_SpecCube0_ProbePosition, Vector4.zero);
                }
                computeShader.SetFloat(_ProbeWeight, 0.0f);
                computeShader.SetFloat(_ProbeSet, 1.0f);
            }
            else
            {
                computeShader.SetFloat(_ProbeSet, 0.0f);
            }
        }

        private static ReflectionProbe GetClosestProbe(NativeArray<VisibleReflectionProbe> visibleReflectionProbes, Vector3 cameraPosition)
        {
            ReflectionProbe closestProbe = null;
            float closestDistance = float.MaxValue;
            int highestImportance = int.MinValue;
            float smallestBoundsSize = float.MaxValue;

            foreach (var visibleProbe in visibleReflectionProbes)
            {
                ReflectionProbe probe = visibleProbe.reflectionProbe;
                if (probe == null) continue;

                Bounds probeBounds = probe.bounds;
                int probeImportance = probe.importance;
                float boundsSize = probeBounds.size.magnitude;

                if (probeBounds.Contains(cameraPosition))
                {
                    float distance = Vector3.Distance(cameraPosition, probe.transform.position);

                    bool isMoreImportant = probeImportance > highestImportance;
                    bool isSizeSmaller = probeImportance == highestImportance && boundsSize < smallestBoundsSize;
                    bool isDistanceCloser = boundsSize == smallestBoundsSize && distance < closestDistance;

                    if (isMoreImportant || isSizeSmaller || isDistanceCloser)
                    {
                        closestDistance = distance;
                        highestImportance = probeImportance;
                        smallestBoundsSize = boundsSize;
                        closestProbe = probe;
                    }
                }
            }
            return closestProbe;
        }

        private Vector4 EvaluateRotator(float rand)
        {
            float ca = Mathf.Cos(rand);
            float sa = Mathf.Sin(rand);
            return new Vector4(ca, sa, -sa, ca);
        }

        public void Dispose()
        {
            m_IntermediateCameraColorHandle?.Release();
            m_CombinedOutputHandle?.Release();
            m_DiffuseHandle?.Release();
            m_IntermediateDiffuseHandle?.Release();
            m_AccumulateSampleHandle?.Release();
            m_APVLightingHandle?.Release();
            m_HistoryDepthHandle?.Release();
            m_HistoryCameraColorHandle?.Release();
            m_HistoryIndirectDiffuseHandle?.Release();
            m_AccumulateHistorySampleHandle?.Release();

            for (int i = 0; i < MAX_CAMERA_COUNT; i++)
            {
                cameraHistoryData[i].historyDepthHandle?.Release();
                cameraHistoryData[i].historyCameraColorHandle?.Release();
                cameraHistoryData[i].historyIndirectDiffuseHandle?.Release();
                cameraHistoryData[i].accumulateHistorySampleHandle?.Release();
            }
        }
        #endregion
    }

    /// <summary>
    /// Backface data render pass
    /// </summary>
    public class BackfaceDataPass : ScriptableRenderPass
    {
        private const string m_ProfilerTag = "Render Backface Data";
        private readonly ProfilingSampler m_ProfilingSampler = new ProfilingSampler(m_ProfilerTag);

        private RTHandle m_BackDepthHandle;
        private RTHandle m_BackColorHandle;
        public bool backfaceLighting;

        private RenderStateBlock m_RenderStateBlock = new RenderStateBlock(RenderStateMask.Nothing);
        private readonly ShaderTagId[] m_LitTags = new ShaderTagId[2];

        private const string k_DepthOnly = "DepthOnly";
        private const string k_UniversalForward = "UniversalForward";
        private const string k_UniversalForwardOnly = "UniversalForwardOnly";

        private static readonly int _CameraBackDepthTexture = Shader.PropertyToID("_CameraBackDepthTexture");
        private static readonly int _CameraBackOpaqueTexture = Shader.PropertyToID("_CameraBackOpaqueTexture");

#if UNITY_6000_0_OR_NEWER
        [Obsolete]
#endif
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            var depthDesc = renderingData.cameraData.cameraTargetDescriptor;
            depthDesc.msaaSamples = 1;
            depthDesc.bindMS = false;
            depthDesc.graphicsFormat = GraphicsFormat.None;

            if (!backfaceLighting)
            {
#if UNITY_6000_0_OR_NEWER
                RenderingUtils.ReAllocateHandleIfNeeded(ref m_BackDepthHandle, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_CameraBackDepthTexture");
#else
                RenderingUtils.ReAllocateIfNeeded(ref m_BackDepthHandle, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_CameraBackDepthTexture");
#endif
                cmd.SetGlobalTexture(_CameraBackDepthTexture, m_BackDepthHandle);
                ConfigureTarget(m_BackDepthHandle, m_BackDepthHandle);
                ConfigureClear(ClearFlag.Depth, Color.clear);
            }
            else
            {
                var colorDesc = renderingData.cameraData.cameraTargetDescriptor;
                colorDesc.depthStencilFormat = GraphicsFormat.None;
                colorDesc.msaaSamples = 1;
                colorDesc.bindMS = false;
                colorDesc.graphicsFormat = GraphicsFormat.B10G11R11_UFloatPack32;

#if UNITY_6000_0_OR_NEWER
                RenderingUtils.ReAllocateHandleIfNeeded(ref m_BackColorHandle, colorDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_CameraBackOpaqueTexture");
                RenderingUtils.ReAllocateHandleIfNeeded(ref m_BackDepthHandle, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_CameraBackDepthTexture");
#else
                RenderingUtils.ReAllocateIfNeeded(ref m_BackColorHandle, colorDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_CameraBackOpaqueTexture");
                RenderingUtils.ReAllocateIfNeeded(ref m_BackDepthHandle, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_CameraBackDepthTexture");
#endif

                cmd.SetGlobalTexture(_CameraBackDepthTexture, m_BackDepthHandle);
                cmd.SetGlobalTexture(_CameraBackOpaqueTexture, m_BackColorHandle);
                ConfigureTarget(m_BackColorHandle, m_BackDepthHandle);
                ConfigureClear(ClearFlag.Color | ClearFlag.Depth, Color.clear);
            }
        }

#if UNITY_6000_0_OR_NEWER
        [Obsolete]
#endif
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get();

            using (new ProfilingScope(cmd, m_ProfilingSampler))
            {
                ShaderTagId passId = backfaceLighting ? 
                    new ShaderTagId(k_UniversalForward) : new ShaderTagId(k_DepthOnly);

                if (backfaceLighting)
                {
                    m_LitTags[0] = new ShaderTagId(k_UniversalForward);
                    m_LitTags[1] = new ShaderTagId(k_UniversalForwardOnly);
                }

                RendererListDesc rendererListDesc = new RendererListDesc(
                    backfaceLighting ? m_LitTags : new ShaderTagId[] { passId },
                    renderingData.cullResults,
                    renderingData.cameraData.camera);

                m_RenderStateBlock.depthState = new DepthState(true, CompareFunction.LessEqual);
                m_RenderStateBlock.mask |= RenderStateMask.Depth;
                m_RenderStateBlock.rasterState = new RasterState(CullMode.Front);
                m_RenderStateBlock.mask |= RenderStateMask.Raster;
                rendererListDesc.stateBlock = m_RenderStateBlock;
                rendererListDesc.sortingCriteria = renderingData.cameraData.defaultOpaqueSortFlags;
                rendererListDesc.renderQueueRange = RenderQueueRange.opaque;

                RendererList rendererList = context.CreateRendererList(rendererListDesc);
                cmd.DrawRendererList(rendererList);
            }

            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);
        }

#if UNITY_6000_0_OR_NEWER
        private class PassData
        {
            internal RendererListHandle rendererListHandle;
        }

        static void ExecutePass(PassData data, RasterGraphContext context)
        {
            context.cmd.DrawRendererList(data.rendererListHandle);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using (var builder = renderGraph.AddRasterRenderPass<PassData>(m_ProfilerTag, out var passData))
            {
                UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
                UniversalRenderingData universalRenderingData = frameData.Get<UniversalRenderingData>();
                UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();

                TextureDesc depthDesc;
                if (!resourceData.isActiveTargetBackBuffer)
                {
                    depthDesc = resourceData.activeDepthTexture.GetDescriptor(renderGraph);
                }
                else
                {
                    depthDesc = resourceData.cameraDepthTexture.GetDescriptor(renderGraph);
                    var backBufferInfo = renderGraph.GetRenderTargetInfo(resourceData.backBufferDepth);
                    depthDesc.colorFormat = backBufferInfo.format;
                }
                depthDesc.name = "_CameraBackDepthTexture";
                depthDesc.useMipMap = false;
                depthDesc.clearBuffer = true;
                depthDesc.msaaSamples = MSAASamples.None;
                depthDesc.bindTextureMS = false;
                depthDesc.filterMode = FilterMode.Point;
                depthDesc.wrapMode = TextureWrapMode.Clamp;

                if (!backfaceLighting)
                {
                    TextureHandle backDepthHandle = renderGraph.CreateTexture(depthDesc);

                    RendererListDesc rendererListDesc = new RendererListDesc(
                        new ShaderTagId(k_DepthOnly),
                        universalRenderingData.cullResults,
                        cameraData.camera);

                    m_RenderStateBlock.depthState = new DepthState(true, CompareFunction.LessEqual);
                    m_RenderStateBlock.mask |= RenderStateMask.Depth;
                    m_RenderStateBlock.rasterState = new RasterState(CullMode.Front);
                    m_RenderStateBlock.mask |= RenderStateMask.Raster;
                    rendererListDesc.stateBlock = m_RenderStateBlock;
                    rendererListDesc.sortingCriteria = cameraData.defaultOpaqueSortFlags;
                    rendererListDesc.renderQueueRange = RenderQueueRange.opaque;

                    passData.rendererListHandle = renderGraph.CreateRendererList(rendererListDesc);

                    builder.UseRendererList(passData.rendererListHandle);
                    builder.SetRenderAttachmentDepth(backDepthHandle, AccessFlags.ReadWrite);
                    builder.SetGlobalTextureAfterPass(backDepthHandle, _CameraBackDepthTexture);

                    builder.SetRenderFunc((PassData data, RasterGraphContext context) => ExecutePass(data, context));
                }
                else
                {
                    TextureHandle backDepthHandle = renderGraph.CreateTexture(depthDesc);

                    var colorDesc = resourceData.cameraColor.GetDescriptor(renderGraph);
                    colorDesc.name = "_CameraBackOpaqueTexture";
                    colorDesc.useMipMap = false;
                    colorDesc.clearBuffer = true;
                    colorDesc.msaaSamples = MSAASamples.None;
                    colorDesc.bindTextureMS = false;
                    colorDesc.filterMode = FilterMode.Point;
                    colorDesc.wrapMode = TextureWrapMode.Clamp;
                    colorDesc.colorFormat = GraphicsFormat.B10G11R11_UFloatPack32;

                    TextureHandle backColorHandle = renderGraph.CreateTexture(colorDesc);

                    m_LitTags[0] = new ShaderTagId(k_UniversalForward);
                    m_LitTags[1] = new ShaderTagId(k_UniversalForwardOnly);

                    RendererListDesc rendererListDesc = new RendererListDesc(
                        m_LitTags,
                        universalRenderingData.cullResults,
                        cameraData.camera);

                    m_RenderStateBlock.depthState = new DepthState(true, CompareFunction.LessEqual);
                    m_RenderStateBlock.mask |= RenderStateMask.Depth;
                    m_RenderStateBlock.rasterState = new RasterState(CullMode.Front);
                    m_RenderStateBlock.mask |= RenderStateMask.Raster;
                    rendererListDesc.stateBlock = m_RenderStateBlock;
                    rendererListDesc.sortingCriteria = cameraData.defaultOpaqueSortFlags;
                    rendererListDesc.renderQueueRange = RenderQueueRange.opaque;

                    passData.rendererListHandle = renderGraph.CreateRendererList(rendererListDesc);

                    builder.UseRendererList(passData.rendererListHandle);
                    builder.SetRenderAttachment(backColorHandle, 0);
                    builder.SetRenderAttachmentDepth(backDepthHandle);
                    builder.SetGlobalTextureAfterPass(backColorHandle, _CameraBackOpaqueTexture);
                    builder.SetGlobalTextureAfterPass(backDepthHandle, _CameraBackDepthTexture);

                    builder.SetRenderFunc((PassData data, RasterGraphContext context) => ExecutePass(data, context));
                }
            }
        }
#endif

        public void Dispose()
        {
            m_BackDepthHandle?.Release();
            m_BackColorHandle?.Release();
        }
    }

    /// <summary>
    /// Forward GBuffer render pass
    /// </summary>
    public class ForwardGBufferPass : ScriptableRenderPass
    {
        private const string m_ProfilerTag = "Render Forward GBuffer";
        private readonly ProfilingSampler m_ProfilingSampler = new ProfilingSampler(m_ProfilerTag);

        private List<ShaderTagId> m_ShaderTagIdList = new List<ShaderTagId>();
        private FilteringSettings m_filter;
        private RenderStateBlock m_RenderStateBlock = new RenderStateBlock(RenderStateMask.Nothing);

        public RTHandle m_GBuffer0;
        public RTHandle m_GBuffer1;
        public RTHandle m_GBuffer2;
        public RTHandle m_GBufferDepth;
        private RTHandle[] m_GBuffers;

        private static readonly int _GBuffer0_ID = Shader.PropertyToID("_GBuffer0");
        private static readonly int _GBuffer1_ID = Shader.PropertyToID("_GBuffer1");
        private static readonly int _GBuffer2_ID = Shader.PropertyToID("_GBuffer2");

        public ForwardGBufferPass(string[] PassNames)
        {
            RenderQueueRange queue = RenderQueueRange.opaque;
            m_filter = new FilteringSettings(queue);
            if (PassNames != null && PassNames.Length > 0)
            {
                foreach (var passName in PassNames)
                    m_ShaderTagIdList.Add(new ShaderTagId(passName));
            }
        }

        public GraphicsFormat GetGBufferFormat(int index)
        {
            if (index == 0)
                return QualitySettings.activeColorSpace == ColorSpace.Linear ? GraphicsFormat.R8G8B8A8_SRGB : GraphicsFormat.R8G8B8A8_UNorm;
            else if (index == 1)
                return GraphicsFormat.R8G8B8A8_UNorm;
            else if (index == 2)
            {
#if UNITY_2023_2_OR_NEWER
                if (SystemInfo.IsFormatSupported(GraphicsFormat.R8G8B8A8_SNorm, GraphicsFormatUsage.Render))
#else
                if (SystemInfo.IsFormatSupported(GraphicsFormat.R8G8B8A8_SNorm, FormatUsage.Render))
#endif
                    return GraphicsFormat.R8G8B8A8_SNorm;
                else
                    return GraphicsFormat.R16G16B16A16_SFloat;
            }
            else
                return GraphicsFormat.None;
        }

#if UNITY_6000_0_OR_NEWER
        [Obsolete]
#endif
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            RenderTextureDescriptor desc = renderingData.cameraData.cameraTargetDescriptor;
            desc.depthBufferBits = 0;
            desc.stencilFormat = GraphicsFormat.None;
            desc.msaaSamples = 1;
            desc.bindMS = false;

            desc.graphicsFormat = GetGBufferFormat(0);
#if UNITY_6000_0_OR_NEWER
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_GBuffer0, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBuffer0");
#else
            RenderingUtils.ReAllocateIfNeeded(ref m_GBuffer0, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBuffer0");
#endif
            cmd.SetGlobalTexture(_GBuffer0_ID, m_GBuffer0);

            desc.graphicsFormat = GetGBufferFormat(1);
#if UNITY_6000_0_OR_NEWER
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_GBuffer1, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBuffer1");
#else
            RenderingUtils.ReAllocateIfNeeded(ref m_GBuffer1, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBuffer1");
#endif
            cmd.SetGlobalTexture(_GBuffer1_ID, m_GBuffer1);

            desc.graphicsFormat = GetGBufferFormat(2);
#if UNITY_6000_0_OR_NEWER
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_GBuffer2, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBuffer2");
#else
            RenderingUtils.ReAllocateIfNeeded(ref m_GBuffer2, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBuffer2");
#endif
            cmd.SetGlobalTexture(_GBuffer2_ID, m_GBuffer2);
            m_GBuffers = new RTHandle[] { m_GBuffer0, m_GBuffer1, m_GBuffer2 };

            bool isOpenGL = (SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLES3) || 
                            (SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLCore);

            bool canDepthPriming = !isOpenGL && 
                (renderingData.cameraData.renderType == CameraRenderType.Base || renderingData.cameraData.clearDepth) && 
                renderingData.cameraData.cameraTargetDescriptor.msaaSamples == desc.msaaSamples;

            RenderTextureDescriptor depthDesc = renderingData.cameraData.cameraTargetDescriptor;
            depthDesc.msaaSamples = 1;
            depthDesc.bindMS = false;
            depthDesc.graphicsFormat = GraphicsFormat.None;

#if UNITY_6000_0_OR_NEWER
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_GBufferDepth, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBufferDepthTexture");
#else
            RenderingUtils.ReAllocateIfNeeded(ref m_GBufferDepth, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBufferDepthTexture");
#endif

            if (canDepthPriming)
                ConfigureTarget(m_GBuffers, renderingData.cameraData.renderer.cameraDepthTargetHandle);
            else
                ConfigureTarget(m_GBuffers, m_GBufferDepth);

            ConfigureInput(ScriptableRenderPassInput.Depth);

            if (isOpenGL)
                ConfigureClear(ClearFlag.Color | ClearFlag.Depth, Color.black);
            else
                ConfigureClear(ClearFlag.Color, Color.clear);

            if (canDepthPriming)
            {
                m_RenderStateBlock.depthState = new DepthState(false, CompareFunction.Equal);
                m_RenderStateBlock.mask |= RenderStateMask.Depth;
            }
            else if (m_RenderStateBlock.depthState.compareFunction == CompareFunction.Equal)
            {
                m_RenderStateBlock.depthState = new DepthState(true, CompareFunction.LessEqual);
                m_RenderStateBlock.mask |= RenderStateMask.Depth;
            }
        }

#if UNITY_6000_0_OR_NEWER
        [Obsolete]
#endif
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            SortingCriteria sortingCriteria = renderingData.cameraData.defaultOpaqueSortFlags;

            CommandBuffer cmd = CommandBufferPool.Get();
            using (new ProfilingScope(cmd, m_ProfilingSampler))
            {
                RendererListDesc rendererListDesc = new RendererListDesc(m_ShaderTagIdList[0], renderingData.cullResults, renderingData.cameraData.camera);
                rendererListDesc.stateBlock = m_RenderStateBlock;
                rendererListDesc.sortingCriteria = sortingCriteria;
                rendererListDesc.renderQueueRange = m_filter.renderQueueRange;
                RendererList rendererList = context.CreateRendererList(rendererListDesc);

                cmd.DrawRendererList(rendererList);
            }

            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);
        }

#if UNITY_6000_0_OR_NEWER
        private class PassData
        {
            internal bool isOpenGL;
            internal RendererListHandle rendererListHandle;
        }

        static void ExecutePass(PassData data, RasterGraphContext context)
        {
            if (data.isOpenGL)
                context.cmd.ClearRenderTarget(true, true, Color.black);

            context.cmd.DrawRendererList(data.rendererListHandle);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using (var builder = renderGraph.AddRasterRenderPass<PassData>(m_ProfilerTag, out var passData))
            {
                UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
                UniversalRenderingData universalRenderingData = frameData.Get<UniversalRenderingData>();
                UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
                UniversalLightData lightData = frameData.Get<UniversalLightData>();

                RenderTextureDescriptor desc = cameraData.cameraTargetDescriptor;
                desc.msaaSamples = 1;
                desc.bindMS = false;
                desc.depthBufferBits = 0;

                desc.graphicsFormat = GetGBufferFormat(0);
                TextureHandle gBuffer0Handle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, desc, "_GBuffer0", false, FilterMode.Point, TextureWrapMode.Clamp);

                desc.graphicsFormat = GetGBufferFormat(1);
                TextureHandle gBuffer1Handle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, desc, "_GBuffer1", false, FilterMode.Point, TextureWrapMode.Clamp);

                desc.graphicsFormat = GetGBufferFormat(2);
                TextureHandle gBuffer2Handle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, desc, "_GBuffer2", false, FilterMode.Point, TextureWrapMode.Clamp);

                bool isOpenGL = (SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLES3) || 
                                (SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLCore);

                bool canDepthPriming = !isOpenGL && 
                    (cameraData.renderType == CameraRenderType.Base || cameraData.clearDepth) && 
                    cameraData.cameraTargetDescriptor.msaaSamples == desc.msaaSamples;

                TextureDesc depthDesc;
                if (!resourceData.isActiveTargetBackBuffer)
                {
                    depthDesc = resourceData.activeDepthTexture.GetDescriptor(renderGraph);
                }
                else
                {
                    depthDesc = resourceData.cameraDepthTexture.GetDescriptor(renderGraph);
                    var backBufferInfo = renderGraph.GetRenderTargetInfo(resourceData.backBufferDepth);
                    depthDesc.colorFormat = backBufferInfo.format;
                }
                depthDesc.name = "_GBufferDepthTexture";
                depthDesc.useMipMap = false;
                depthDesc.clearBuffer = false;
                depthDesc.msaaSamples = MSAASamples.None;
                depthDesc.bindTextureMS = false;
                depthDesc.filterMode = FilterMode.Point;
                depthDesc.wrapMode = TextureWrapMode.Clamp;

                TextureHandle depthHandle;
                if (canDepthPriming)
                    depthHandle = resourceData.activeDepthTexture;
                else
                    depthHandle = renderGraph.CreateTexture(depthDesc);

                if (canDepthPriming)
                {
                    m_RenderStateBlock.depthState = new DepthState(false, CompareFunction.Equal);
                    m_RenderStateBlock.mask |= RenderStateMask.Depth;
                }
                else if (m_RenderStateBlock.depthState.compareFunction == CompareFunction.Equal)
                {
                    m_RenderStateBlock.depthState = new DepthState(true, CompareFunction.LessEqual);
                    m_RenderStateBlock.mask |= RenderStateMask.Depth;
                }

                SortingCriteria sortingCriteria = cameraData.defaultOpaqueSortFlags;
                RendererListDesc rendererListDesc = new RendererListDesc(m_ShaderTagIdList[0], universalRenderingData.cullResults, cameraData.camera);
                rendererListDesc.stateBlock = m_RenderStateBlock;
                rendererListDesc.sortingCriteria = sortingCriteria;
                rendererListDesc.renderQueueRange = m_filter.renderQueueRange;

                passData.isOpenGL = isOpenGL;
                passData.rendererListHandle = renderGraph.CreateRendererList(rendererListDesc);

                builder.UseRendererList(passData.rendererListHandle);

                builder.SetRenderAttachment(gBuffer0Handle, 0);
                builder.SetRenderAttachment(gBuffer1Handle, 1);
                builder.SetRenderAttachment(gBuffer2Handle, 2);
                builder.SetRenderAttachmentDepth(depthHandle, AccessFlags.Write);

                builder.SetGlobalTextureAfterPass(gBuffer0Handle, _GBuffer0_ID);
                builder.SetGlobalTextureAfterPass(gBuffer1Handle, _GBuffer1_ID);
                builder.SetGlobalTextureAfterPass(gBuffer2Handle, _GBuffer2_ID);

                builder.SetRenderFunc((PassData data, RasterGraphContext context) => ExecutePass(data, context));
            }
        }
#endif

        public void Dispose()
        {
            m_GBuffer0?.Release();
            m_GBuffer1?.Release();
            m_GBuffer2?.Release();
            m_GBufferDepth?.Release();
        }
    }

    #endregion
}
