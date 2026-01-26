using System;
using System.Reflection;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Experimental.Rendering;

#if UNITY_6000_0_OR_NEWER
using UnityEngine.Rendering.RenderGraphModule;
#endif

[DisallowMultipleRendererFeature("Screen Space Global Illumination (Compute)")]
[Tooltip("Screen Space Global Illumination using Compute Shaders with HZB acceleration.")]
[HelpURL("https://github.com/jiaozi158/UnitySSGIURP")]
public class ScreenSpaceGlobalIlluminationURP : ScriptableRendererFeature
{
    #region Serialized Fields
    [Header("Compute Shaders")]
    [SerializeField] private ComputeShader m_DepthPyramidCS;
    [SerializeField] private ComputeShader m_GBufferCS;
    [SerializeField] private ComputeShader m_RayMarchingCS;
    [SerializeField] private ComputeShader m_TemporalFilterCS;
    [SerializeField] private ComputeShader m_SpatialDenoiseCS;
    [SerializeField] private ComputeShader m_CombineCS;
    [SerializeField] private ComputeShader m_HistoryUpdateCS;

    [Header("Setup")]
    [SerializeField] private bool m_RenderingDebugger = false;

    [Header("Performance")]
    [SerializeField] private bool m_ReflectionProbes = true;
    [SerializeField] private bool m_HighQualityUpscaling = false;

    [Header("Lighting")]
    [SerializeField] private bool m_OverrideAmbientLighting = true;

    [Header("Advanced")]
    [SerializeField] private bool m_BackfaceLighting = false;
    #endregion

    #region Public Properties
    public bool RenderingDebugger
    {
        get => m_RenderingDebugger;
        set => m_RenderingDebugger = value;
    }

    public bool ReflectionProbes
    {
        get => m_ReflectionProbes;
        set => m_ReflectionProbes = value;
    }

    public bool HighQualityUpscaling
    {
        get => m_HighQualityUpscaling;
        set => m_HighQualityUpscaling = value;
    }

    public bool OverrideAmbientLighting
    {
        get => m_OverrideAmbientLighting;
        set => m_OverrideAmbientLighting = value;
    }

    public bool BackfaceLighting
    {
        get => m_BackfaceLighting;
        set => m_BackfaceLighting = value;
    }
    #endregion

    #region Private Fields
    private DepthPyramidPass m_DepthPyramidPass;
    private SSGIGBufferPass m_GBufferPass;
    private SSGIMainPass m_SSGIPass;
    private BackfaceDataPass m_BackfaceDataPass;
    private ForwardGBufferPass m_ForwardGBufferPass;

    // Reflection for checking deferred renderer
    private static readonly FieldInfo s_GBufferFieldInfo = 
        typeof(UniversalRenderer).GetField("m_GBufferPass", BindingFlags.NonPublic | BindingFlags.Instance);

    // Log flags
    private bool m_IsDebuggerLogPrinted = false;
    private bool m_IsBackfaceLightingLogPrinted = false;
    private bool m_IsMissingComputeShaderLogPrinted = false;

    // Global keywords
    private const string SSGI_RENDER_GBUFFER = "SSGI_RENDER_GBUFFER";
    private const string SSGI_RENDER_BACKFACE_DEPTH = "SSGI_RENDER_BACKFACE_DEPTH";
    private const string SSGI_RENDER_BACKFACE_COLOR = "SSGI_RENDER_BACKFACE_COLOR";
    #endregion

    #region Lifecycle
    public override void Create()
    {
        if (!ValidateComputeShaders())
        {
            return;
        }

        m_DepthPyramidPass = new DepthPyramidPass(m_DepthPyramidCS);
        m_DepthPyramidPass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;

        m_GBufferPass = new SSGIGBufferPass(m_GBufferCS);
        m_GBufferPass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;

        m_SSGIPass = new SSGIMainPass(
            m_RayMarchingCS,
            m_TemporalFilterCS,
            m_SpatialDenoiseCS,
            m_CombineCS,
            m_HistoryUpdateCS
        );
    #if UNITY_6000_0_OR_NEWER
        bool enableRenderGraph = !GraphicsSettings.GetRenderPipelineSettings<RenderGraphSettings>().enableRenderCompatibilityMode;
        m_SSGIPass.renderPassEvent = enableRenderGraph ? RenderPassEvent.AfterRenderingSkybox : RenderPassEvent.BeforeRenderingTransparents;
    #else
        m_SSGIPass.renderPassEvent = RenderPassEvent.BeforeRenderingTransparents;
    #endif

        m_BackfaceDataPass = new BackfaceDataPass();
        m_BackfaceDataPass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques - 1;

        m_ForwardGBufferPass = new ForwardGBufferPass();
        m_ForwardGBufferPass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;
    }

    protected override void Dispose(bool disposing)
    {
        m_DepthPyramidPass?.Dispose();
        m_GBufferPass?.Dispose();
        m_SSGIPass?.Dispose();
        m_BackfaceDataPass?.Dispose();
        m_ForwardGBufferPass?.Dispose();
    }

    private bool ValidateComputeShaders()
    {
        bool valid = m_DepthPyramidCS != null &&
                     m_GBufferCS != null &&
                     m_RayMarchingCS != null &&
                     m_TemporalFilterCS != null &&
                     m_SpatialDenoiseCS != null &&
                     m_CombineCS != null &&
                     m_HistoryUpdateCS != null;

        if (!valid && !m_IsMissingComputeShaderLogPrinted)
        {
            Debug.LogError("SSGI: One or more compute shaders are missing. Please assign all compute shaders in the renderer feature.");
            m_IsMissingComputeShaderLogPrinted = true;
        }

        return valid;
    }
    #endregion

    #region AddRenderPasses
    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        if (!ValidateComputeShaders())
            return;

        var camera = renderingData.cameraData.camera;
        if (camera.cameraType == CameraType.Preview)
            return;

        var stack = VolumeManager.instance.stack;
        var ssgiVolume = stack.GetComponent<ScreenSpaceGlobalIlluminationVolume>();

        if (ssgiVolume == null || !ssgiVolume.IsActive())
            return;

        // Check conditions to skip rendering
        bool shouldDisable = !m_ReflectionProbes && camera.cameraType == CameraType.Reflection;
        shouldDisable |= ssgiVolume.indirectDiffuseLightingMultiplier.value == 0.0f && !m_OverrideAmbientLighting;
        shouldDisable |= renderingData.cameraData.renderType == CameraRenderType.Overlay;

        if (shouldDisable)
            return;

        // Check rendering debugger
        bool isDebugger = DebugManager.instance.isAnyDebugUIActive;
        if (isDebugger && !m_RenderingDebugger)
        {
        #if UNITY_EDITOR || DEBUG
            if (!m_IsDebuggerLogPrinted)
            {
                Debug.Log("SSGI: Disabled to avoid affecting rendering debugging.");
                m_IsDebuggerLogPrinted = true;
            }
        #endif
            return;
        }
        m_IsDebuggerLogPrinted = false;

        // Setup pass parameters
        SetupPassParameters(ssgiVolume, ref renderingData);

        // Check rendering path
        bool isUsingDeferred = s_GBufferFieldInfo?.GetValue(renderer) != null;
        isUsingDeferred &= SystemInfo.graphicsDeviceType != GraphicsDeviceType.OpenGLES3;
        isUsingDeferred &= SystemInfo.graphicsDeviceType != GraphicsDeviceType.OpenGLCore;

        // Enqueue passes
        renderer.EnqueuePass(m_DepthPyramidPass);

        bool renderBackfaceData = ssgiVolume.thicknessMode.value != ScreenSpaceGlobalIlluminationVolume.ThicknessMode.Constant;
        if (renderBackfaceData)
        {
            bool supportBackfaceLighting = m_BackfaceLighting && !isUsingDeferred;
            m_BackfaceDataPass.BackfaceLighting = supportBackfaceLighting;
            renderer.EnqueuePass(m_BackfaceDataPass);

            Shader.EnableKeyword(SSGI_RENDER_BACKFACE_DEPTH);
            if (supportBackfaceLighting)
                Shader.EnableKeyword(SSGI_RENDER_BACKFACE_COLOR);
            else
                Shader.DisableKeyword(SSGI_RENDER_BACKFACE_COLOR);

        #if UNITY_EDITOR || DEBUG
            if (m_BackfaceLighting && isUsingDeferred && !m_IsBackfaceLightingLogPrinted)
            {
                Debug.LogError("SSGI: Backface Lighting is only supported on Forward(+) rendering path.");
                m_IsBackfaceLightingLogPrinted = true;
            }
            else
                m_IsBackfaceLightingLogPrinted = false;
        #endif
        }
        else
        {
            Shader.DisableKeyword(SSGI_RENDER_BACKFACE_DEPTH);
            Shader.DisableKeyword(SSGI_RENDER_BACKFACE_COLOR);
        }

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

        // GBuffer processing pass
        renderer.EnqueuePass(m_GBufferPass);

        // Main SSGI pass
        if (!isDebugger || m_RenderingDebugger)
        {
            renderer.EnqueuePass(m_SSGIPass);
        }
    }

    private void SetupPassParameters(ScreenSpaceGlobalIlluminationVolume volume, ref RenderingData renderingData)
    {
        var camera = renderingData.cameraData.camera;

        // Calculate step parameters
        bool lowStepCount = volume.maxRaySteps.value <= 16;
        int groupsCount = volume.maxRaySteps.value / 8;
        int smallSteps = lowStepCount ? 0 : Mathf.Max(groupsCount, 4);
        int mediumSteps = lowStepCount ? groupsCount + 2 : smallSteps + groupsCount * 2;

        float resolutionScale = volume.fullResolutionSS.value ? 1.0f : volume.resolutionScaleSS.value;
        float temporalIntensity = Mathf.Lerp(
            volume.denoiseIntensitySS.value + 0.02f,
            volume.denoiseIntensitySS.value - 0.04f,
            resolutionScale
        );

        // Create parameters struct
        var parameters = new SSGIParameters
        {
            maxSteps = volume.maxRaySteps.value,
            maxSmallSteps = smallSteps,
            maxMediumSteps = mediumSteps,
            stepSize = lowStepCount ? 0.5f : 0.4f,
            smallStepSize = smallSteps < 4 ? 0.05f : 0.015f,
            mediumStepSize = lowStepCount ? 0.1f : 0.05f,
            thickness = volume.depthBufferThickness.value,
            thicknessIncrement = volume.depthBufferThickness.value * 0.25f,
            rayCount = volume.sampleCount.value,
            temporalIntensity = temporalIntensity,
            maxBrightness = 7.0f,
            downSample = resolutionScale,
            indirectDiffuseLightingMultiplier = volume.indirectDiffuseLightingMultiplier.value,
            aggressiveDenoise = volume.denoiserAlgorithmSS.value == ScreenSpaceGlobalIlluminationVolume.DenoiserAlgorithm.Aggressive ? 1.0f : 0.0f,
            reBlurDenoiserRadius = volume.denoiserRadiusSS.value * 0.08f,
            isProbeCamera = camera.cameraType == CameraType.Reflection ? 1.0f : 0.0f,
            backDepthEnabled = volume.thicknessMode.value != ScreenSpaceGlobalIlluminationVolume.ThicknessMode.Constant 
                ? (m_BackfaceLighting ? 2.0f : 1.0f) : 0.0f,
            overrideAmbientLighting = m_OverrideAmbientLighting,
            highQualityUpscaling = m_HighQualityUpscaling,
            enableDenoise = volume.denoiseSS.value,
            secondDenoisePass = volume.secondDenoiserPassSS.value,
            fallbackSky = volume.IsFallbackSky(),
            fallbackReflectionProbes = volume.IsFallbackReflectionProbes()
        };

    #if UNITY_2023_3_OR_NEWER
        parameters.indirectDiffuseRenderingLayers = (uint)volume.indirectDiffuseRenderingLayers.value.value;
        parameters.enableRenderingLayers = Shader.IsKeywordEnabled("_WRITE_RENDERING_LAYERS") && 
                                           volume.indirectDiffuseRenderingLayers.value.value != 0xFFFF;
    #else
        parameters.indirectDiffuseRenderingLayers = 0xFFFFFFFF;
        parameters.enableRenderingLayers = false;
    #endif

        // Pass parameters to passes
        m_GBufferPass.Parameters = parameters;
        m_SSGIPass.Parameters = parameters;
        m_SSGIPass.Volume = volume;
    }
    #endregion
}

#region Parameters Struct
[Serializable]
public struct SSGIParameters
{
    public float maxSteps;
    public float maxSmallSteps;
    public float maxMediumSteps;
    public float stepSize;
    public float smallStepSize;
    public float mediumStepSize;
    public float thickness;
    public float thicknessIncrement;
    public float rayCount;
    public float temporalIntensity;
    public float maxBrightness;
    public float downSample;
    public float indirectDiffuseLightingMultiplier;
    public uint indirectDiffuseRenderingLayers;
    public float aggressiveDenoise;
    public float reBlurDenoiserRadius;
    public float isProbeCamera;
    public float backDepthEnabled;
    public bool overrideAmbientLighting;
    public bool highQualityUpscaling;
    public bool enableDenoise;
    public bool secondDenoisePass;
    public bool enableRenderingLayers;
    public bool fallbackSky;
    public bool fallbackReflectionProbes;
}
#endregion
