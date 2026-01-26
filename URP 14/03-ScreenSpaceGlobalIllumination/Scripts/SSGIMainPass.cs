using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Experimental.Rendering;
using Unity.Collections;

#if UNITY_6000_0_OR_NEWER
using UnityEngine.Rendering.RenderGraphModule;
#endif

public class SSGIMainPass : ScriptableRenderPass
{
    private const string k_ProfilerTag = "Screen Space Global Illumination";
    private readonly ProfilingSampler m_ProfilingSampler = new ProfilingSampler(k_ProfilerTag);

    // Compute shaders
    private readonly ComputeShader m_RayMarchingCS;
    private readonly ComputeShader m_TemporalFilterCS;
    private readonly ComputeShader m_SpatialDenoiseCS;
    private readonly ComputeShader m_CombineCS;
    private readonly ComputeShader m_HistoryUpdateCS;

    // Kernel indices
    private int m_RayMarchingKernel;
    private int m_TemporalReprojectionKernel;
    private int m_TemporalStabilizationKernel;
    private int m_EdgeAwareDenoiseKernel;
    private int m_PoissonDiskDenoiseKernel;
    private int m_CombineGIKernel;
    private int m_CopyHistoryDepthKernel;
    private int m_CopyHistoryColorKernel;
    private int m_UpdateHistoryBuffersKernel;

    // RTHandles
    private RTHandle m_IndirectDiffuseHandle;
    private RTHandle m_IntermediateHandle;
    private RTHandle m_SampleCountHandle;
    private RTHandle m_CombineOutputHandle; // NEW: UAV-compatible output for combine

    // History RTHandles (persistent per camera)
    private const int k_MaxCameraCount = 4;
    private CameraHistoryData[] m_CameraHistoryData;

    private struct CameraHistoryData
    {
        public int hash;
        public RTHandle historyDepthHandle;
        public RTHandle historyCameraColorHandle;
        public RTHandle historyIndirectDiffuseHandle;
        public RTHandle historySampleHandle;
        public Matrix4x4 prevInvViewProjMatrix;
        public Vector3 prevCameraPositionWS;
        public float scaledWidth;
        public float scaledHeight;
        public bool isValid;
    }

    public SSGIParameters Parameters { get; set; }
    public ScreenSpaceGlobalIlluminationVolume Volume { get; set; }

    private int m_FrameCount;
    private int m_CurrentCameraIndex;

    #region Shader Property IDs
    // Texture properties
    private static readonly int s_CameraColorTexture = Shader.PropertyToID("_CameraColorTexture");
    private static readonly int s_CameraDepthTexture = Shader.PropertyToID("_CameraDepthTexture");
    private static readonly int s_MotionVectorTexture = Shader.PropertyToID("_MotionVectorTexture");
    private static readonly int s_GBuffer0 = Shader.PropertyToID("_GBuffer0");
    private static readonly int s_GBuffer1 = Shader.PropertyToID("_GBuffer1");
    private static readonly int s_GBuffer2 = Shader.PropertyToID("_GBuffer2");
    private static readonly int s_CameraBackDepthTexture = Shader.PropertyToID("_CameraBackDepthTexture");
    private static readonly int s_CameraBackOpaqueTexture = Shader.PropertyToID("_CameraBackOpaqueTexture");
    
    private static readonly int s_SSGIOutput = Shader.PropertyToID("_SSGIOutput");
    private static readonly int s_SSGISampleOutput = Shader.PropertyToID("_SSGISampleOutput");
    private static readonly int s_SSGIInputTexture = Shader.PropertyToID("_SSGIInputTexture");
    private static readonly int s_IndirectDiffuseTexture = Shader.PropertyToID("_IndirectDiffuseTexture");
    private static readonly int s_SSGISampleTexture = Shader.PropertyToID("_SSGISampleTexture");
    
    // History textures
    private static readonly int s_SSGIHistoryDepthTexture = Shader.PropertyToID("_SSGIHistoryDepthTexture");
    private static readonly int s_SSGIHistoryCameraColorTexture = Shader.PropertyToID("_SSGIHistoryCameraColorTexture");
    private static readonly int s_HistoryIndirectDiffuseTexture = Shader.PropertyToID("_HistoryIndirectDiffuseTexture");
    private static readonly int s_SSGIHistorySampleTexture = Shader.PropertyToID("_SSGIHistorySampleTexture");
    
    private static readonly int s_HistoryDepthOutput = Shader.PropertyToID("_HistoryDepthOutput");
    private static readonly int s_HistoryColorOutput = Shader.PropertyToID("_HistoryColorOutput");
    private static readonly int s_HistoryIndirectOutput = Shader.PropertyToID("_HistoryIndirectOutput");
    private static readonly int s_HistorySampleOutput = Shader.PropertyToID("_HistorySampleOutput");

    // Parameters
    private static readonly int s_SSGITextureSizes = Shader.PropertyToID("_SSGITextureSizes");
    private static readonly int s_SSGIIndirectTextureSizes = Shader.PropertyToID("_SSGIIndirectTextureSizes");
    private static readonly int s_MaxSteps = Shader.PropertyToID("_MaxSteps");
    private static readonly int s_MaxSmallSteps = Shader.PropertyToID("_MaxSmallSteps");
    private static readonly int s_MaxMediumSteps = Shader.PropertyToID("_MaxMediumSteps");
    private static readonly int s_StepSize = Shader.PropertyToID("_StepSize");
    private static readonly int s_SmallStepSize = Shader.PropertyToID("_SmallStepSize");
    private static readonly int s_MediumStepSize = Shader.PropertyToID("_MediumStepSize");
    private static readonly int s_Thickness = Shader.PropertyToID("_Thickness");
    private static readonly int s_ThicknessIncrement = Shader.PropertyToID("_ThicknessIncrement");
    private static readonly int s_RayCount = Shader.PropertyToID("_RayCount");
    private static readonly int s_TemporalIntensity = Shader.PropertyToID("_TemporalIntensity");
    private static readonly int s_MaxBrightness = Shader.PropertyToID("_MaxBrightness");
    private static readonly int s_DownSample = Shader.PropertyToID("_DownSample");
    private static readonly int s_FrameIndex = Shader.PropertyToID("_FrameIndex");
    private static readonly int s_HistoryTextureValid = Shader.PropertyToID("_HistoryTextureValid");
    private static readonly int s_IsProbeCamera = Shader.PropertyToID("_IsProbeCamera");
    private static readonly int s_BackDepthEnabled = Shader.PropertyToID("_BackDepthEnabled");
    private static readonly int s_IndirectDiffuseLightingMultiplier = Shader.PropertyToID("_IndirectDiffuseLightingMultiplier");
    private static readonly int s_IndirectDiffuseRenderingLayers = Shader.PropertyToID("_IndirectDiffuseRenderingLayers");
    private static readonly int s_AggressiveDenoise = Shader.PropertyToID("_AggressiveDenoise");
    private static readonly int s_ReBlurBlurRotator = Shader.PropertyToID("_ReBlurBlurRotator");
    private static readonly int s_ReBlurDenoiserRadius = Shader.PropertyToID("_ReBlurDenoiserRadius");
    private static readonly int s_PrevInvViewProjMatrix = Shader.PropertyToID("_PrevInvViewProjMatrix");
    private static readonly int s_PrevCameraPositionWS = Shader.PropertyToID("_PrevCameraPositionWS");
    private static readonly int s_PixelSpreadAngleTangent = Shader.PropertyToID("_PixelSpreadAngleTangent");

    // SH coefficients
    private static readonly int s_SSGI_SHAr = Shader.PropertyToID("_SSGI_SHAr");
    private static readonly int s_SSGI_SHAg = Shader.PropertyToID("_SSGI_SHAg");
    private static readonly int s_SSGI_SHAb = Shader.PropertyToID("_SSGI_SHAb");
    private static readonly int s_SSGI_SHBr = Shader.PropertyToID("_SSGI_SHBr");
    private static readonly int s_SSGI_SHBg = Shader.PropertyToID("_SSGI_SHBg");
    private static readonly int s_SSGI_SHBb = Shader.PropertyToID("_SSGI_SHBb");
    private static readonly int s_SSGI_SHC = Shader.PropertyToID("_SSGI_SHC");
    #endregion

    // Blur rotator values (from original shader)
    private static readonly float[] k_BlurRands = new float[] 
    { 
        0.61264f, 0.296032f, 0.637552f, 0.524287f, 0.493583f, 0.972775f, 0.292517f, 0.771358f,
        0.526745f, 0.769914f, 0.400229f, 0.891529f, 0.283315f, 0.352458f, 0.807725f, 0.919026f,
        0.0697553f, 0.949327f, 0.525995f, 0.0860558f, 0.192214f, 0.663227f, 0.890233f, 0.348893f,
        0.0641713f, 0.020023f, 0.457702f, 0.0630958f, 0.23828f, 0.970634f, 0.902208f, 0.85092f 
    };

    public SSGIMainPass(
        ComputeShader rayMarchingCS,
        ComputeShader temporalFilterCS,
        ComputeShader spatialDenoiseCS,
        ComputeShader combineCS,
        ComputeShader historyUpdateCS)
    {
        m_RayMarchingCS = rayMarchingCS;
        m_TemporalFilterCS = temporalFilterCS;
        m_SpatialDenoiseCS = spatialDenoiseCS;
        m_CombineCS = combineCS;
        m_HistoryUpdateCS = historyUpdateCS;

        // Find kernels
        m_RayMarchingKernel = m_RayMarchingCS.FindKernel("CSRayMarching");
        m_TemporalReprojectionKernel = m_TemporalFilterCS.FindKernel("CSTemporalReprojection");
        m_TemporalStabilizationKernel = m_TemporalFilterCS.FindKernel("CSTemporalStabilization");
        m_EdgeAwareDenoiseKernel = m_SpatialDenoiseCS.FindKernel("CSEdgeAwareSpatialDenoise");
        m_PoissonDiskDenoiseKernel = m_SpatialDenoiseCS.FindKernel("CSPoissonDiskDenoise");
        m_CombineGIKernel = m_CombineCS.FindKernel("CSCombineGI");
        m_CopyHistoryDepthKernel = m_CombineCS.FindKernel("CSCopyHistoryDepth");
        m_CopyHistoryColorKernel = m_CombineCS.FindKernel("CSCopyHistoryColor");
        m_UpdateHistoryBuffersKernel = m_HistoryUpdateCS.FindKernel("CSUpdateHistoryBuffers");

        // Initialize camera history
        m_CameraHistoryData = new CameraHistoryData[k_MaxCameraCount];
    }

    private Vector4 EvaluateRotator(float rand)
    {
        float ca = Mathf.Cos(rand);
        float sa = Mathf.Sin(rand);
        return new Vector4(ca, sa, -sa, ca);
    }

    private int GetCameraHistoryIndex(int cameraHash)
    {
        for (int i = 0; i < k_MaxCameraCount; i++)
        {
            if (m_CameraHistoryData[i].hash == cameraHash)
                return i;
        }
        return -1;
    }

    private void UpdateCameraHistory(bool isNewCamera)
    {
        if (isNewCamera)
        {
            // Release oldest camera's history
            int lastIndex = k_MaxCameraCount - 1;
            m_CameraHistoryData[lastIndex].historyDepthHandle?.Release();
            m_CameraHistoryData[lastIndex].historyCameraColorHandle?.Release();
            m_CameraHistoryData[lastIndex].historyIndirectDiffuseHandle?.Release();
            m_CameraHistoryData[lastIndex].historySampleHandle?.Release();

            // Shift array
            Array.Copy(m_CameraHistoryData, 0, m_CameraHistoryData, 1, lastIndex);
            m_CameraHistoryData[0] = new CameraHistoryData();
        }
    }

    #region Non Render Graph
#if UNITY_6000_0_OR_NEWER
    [System.Obsolete]
#endif
    public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
    {
        var camera = renderingData.cameraData.camera;
        var desc = renderingData.cameraData.cameraTargetDescriptor;

        int cameraHash = camera.GetHashCode();
        m_CurrentCameraIndex = GetCameraHistoryIndex(cameraHash);
        bool isNewCamera = m_CurrentCameraIndex == -1;

        UpdateCameraHistory(isNewCamera);
        m_CurrentCameraIndex = isNewCamera ? 0 : m_CurrentCameraIndex;

        ref var historyData = ref m_CameraHistoryData[m_CurrentCameraIndex];
        historyData.hash = cameraHash;

        // Calculate resolution
        float resolutionScale = Parameters.downSample;
        int indirectWidth = Mathf.FloorToInt(desc.width * resolutionScale);
        int indirectHeight = Mathf.FloorToInt(desc.height * resolutionScale);

        // Check if resolution changed
        bool resolutionChanged = historyData.scaledWidth != desc.width || historyData.scaledHeight != desc.height;
        historyData.scaledWidth = desc.width;
        historyData.scaledHeight = desc.height;

        // Allocate indirect diffuse textures
        var indirectDesc = new RenderTextureDescriptor(indirectWidth, indirectHeight, RenderTextureFormat.ARGBHalf, 0)
        {
            enableRandomWrite = true,
            msaaSamples = 1
        };

    #if UNITY_6000_0_OR_NEWER
        RenderingUtils.ReAllocateHandleIfNeeded(ref m_IndirectDiffuseHandle, indirectDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_IndirectDiffuseTexture");
        RenderingUtils.ReAllocateHandleIfNeeded(ref m_IntermediateHandle, indirectDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_IntermediateIndirectTexture");
    #else
        RenderingUtils.ReAllocateIfNeeded(ref m_IndirectDiffuseHandle, indirectDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_IndirectDiffuseTexture");
        RenderingUtils.ReAllocateIfNeeded(ref m_IntermediateHandle, indirectDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_IntermediateIndirectTexture");
    #endif

        // Sample count texture
        var sampleDesc = new RenderTextureDescriptor(indirectWidth, indirectHeight, RenderTextureFormat.RHalf, 0)
        {
            enableRandomWrite = true,
            msaaSamples = 1
        };

    #if UNITY_6000_0_OR_NEWER
        RenderingUtils.ReAllocateHandleIfNeeded(ref m_SampleCountHandle, sampleDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGISampleTexture");
    #else
        RenderingUtils.ReAllocateIfNeeded(ref m_SampleCountHandle, sampleDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGISampleTexture");
    #endif

        // FIX: Allocate UAV-compatible combine output texture
        var combineDesc = new RenderTextureDescriptor(desc.width, desc.height, RenderTextureFormat.ARGBHalf, 0)
        {
            enableRandomWrite = true,
            msaaSamples = 1
        };

    #if UNITY_6000_0_OR_NEWER
        RenderingUtils.ReAllocateHandleIfNeeded(ref m_CombineOutputHandle, combineDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGICombineOutput");
    #else
        RenderingUtils.ReAllocateIfNeeded(ref m_CombineOutputHandle, combineDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGICombineOutput");
    #endif

        // History textures
        var historyColorDesc = new RenderTextureDescriptor(indirectWidth, indirectHeight, RenderTextureFormat.RGB111110Float, 0)
        {
            enableRandomWrite = true,
            msaaSamples = 1
        };

        var historyDepthDesc = new RenderTextureDescriptor(indirectWidth, indirectHeight, RenderTextureFormat.RFloat, 0)
        {
            enableRandomWrite = true,
            msaaSamples = 1
        };

    #if UNITY_6000_0_OR_NEWER
        RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.historyCameraColorHandle, historyColorDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistoryCameraColor");
        RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.historyIndirectDiffuseHandle, indirectDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_HistoryIndirectDiffuse");
        RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.historyDepthHandle, historyDepthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistoryDepth");
        RenderingUtils.ReAllocateHandleIfNeeded(ref historyData.historySampleHandle, sampleDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistorySample");
    #else
        RenderingUtils.ReAllocateIfNeeded(ref historyData.historyCameraColorHandle, historyColorDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistoryCameraColor");
        RenderingUtils.ReAllocateIfNeeded(ref historyData.historyIndirectDiffuseHandle, indirectDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_HistoryIndirectDiffuse");
        RenderingUtils.ReAllocateIfNeeded(ref historyData.historyDepthHandle, historyDepthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistoryDepth");
        RenderingUtils.ReAllocateIfNeeded(ref historyData.historySampleHandle, sampleDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSGIHistorySample");
    #endif

        // Update validity
        if (isNewCamera || resolutionChanged)
            historyData.isValid = false;

        ConfigureInput(ScriptableRenderPassInput.Depth | ScriptableRenderPassInput.Motion);
    }

#if UNITY_6000_0_OR_NEWER
    [System.Obsolete]
#endif
    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        CommandBuffer cmd = CommandBufferPool.Get();

        using (new ProfilingScope(cmd, m_ProfilingSampler))
        {
            var camera = renderingData.cameraData.camera;
            var desc = renderingData.cameraData.cameraTargetDescriptor;
            ref var historyData = ref m_CameraHistoryData[m_CurrentCameraIndex];

            float resolutionScale = Parameters.downSample;
            int indirectWidth = Mathf.FloorToInt(desc.width * resolutionScale);
            int indirectHeight = Mathf.FloorToInt(desc.height * resolutionScale);

            // Update frame counter
            m_FrameCount = (m_FrameCount + 33) % 64000;

            // Set common parameters for all compute shaders
            SetCommonParameters(cmd, camera, desc, indirectWidth, indirectHeight, historyData);

            // 1. Ray Marching
            DispatchRayMarching(cmd, indirectWidth, indirectHeight, historyData);

            if (Parameters.enableDenoise)
            {
                // 2. Temporal Reprojection
                DispatchTemporalReprojection(cmd, indirectWidth, indirectHeight, historyData);

                // 3. Aggressive Denoise (if enabled)
                if (Parameters.aggressiveDenoise > 0)
                {
                    DispatchPoissonDiskDenoise(cmd, indirectWidth, indirectHeight);
                    DispatchPoissonDiskDenoise(cmd, indirectWidth, indirectHeight);
                }

                // 4. Second Denoise Pass (if enabled)
                if (Parameters.secondDenoisePass)
                {
                    DispatchEdgeAwareDenoise(cmd, indirectWidth, indirectHeight);
                    DispatchTemporalStabilization(cmd, indirectWidth, indirectHeight, historyData);
                }

                // Update history buffers
                DispatchHistoryUpdate(cmd, indirectWidth, indirectHeight, historyData);
            }
            else
            {
                // Without denoising, just update history depth
                DispatchCopyHistoryDepth(cmd, indirectWidth, indirectHeight, historyData);
            }

            // 5. Combine with scene color (FIXED: now uses intermediate UAV texture)
            DispatchCombine(cmd, desc.width, desc.height, renderingData.cameraData.renderer.cameraColorTargetHandle);

            // 6. Copy history color
            DispatchCopyHistoryColor(cmd, indirectWidth, indirectHeight, renderingData.cameraData.renderer.cameraColorTargetHandle, historyData);

            // Update previous frame data
            historyData.prevInvViewProjMatrix = (GL.GetGPUProjectionMatrix(camera.projectionMatrix, true) * 
                renderingData.cameraData.GetViewMatrix()).inverse;
            historyData.prevCameraPositionWS = camera.transform.position;
            historyData.isValid = true;
        }

        context.ExecuteCommandBuffer(cmd);
        cmd.Clear();
        CommandBufferPool.Release(cmd);
    }

    private void SetCommonParameters(CommandBuffer cmd, Camera camera, RenderTextureDescriptor desc, 
        int indirectWidth, int indirectHeight, CameraHistoryData historyData)
    {
        Vector4 textureSizes = new Vector4(desc.width, desc.height, 1.0f / desc.width, 1.0f / desc.height);
        Vector4 indirectSizes = new Vector4(indirectWidth, indirectHeight, 1.0f / indirectWidth, 1.0f / indirectHeight);

        // Set for all compute shaders
        ComputeShader[] shaders = { m_RayMarchingCS, m_TemporalFilterCS, m_SpatialDenoiseCS, m_CombineCS, m_HistoryUpdateCS };

        foreach (var cs in shaders)
        {
            cmd.SetComputeVectorParam(cs, s_SSGITextureSizes, textureSizes);
            cmd.SetComputeVectorParam(cs, s_SSGIIndirectTextureSizes, indirectSizes);
            cmd.SetComputeFloatParam(cs, s_MaxSteps, Parameters.maxSteps);
            cmd.SetComputeFloatParam(cs, s_MaxSmallSteps, Parameters.maxSmallSteps);
            cmd.SetComputeFloatParam(cs, s_MaxMediumSteps, Parameters.maxMediumSteps);
            cmd.SetComputeFloatParam(cs, s_StepSize, Parameters.stepSize);
            cmd.SetComputeFloatParam(cs, s_SmallStepSize, Parameters.smallStepSize);
            cmd.SetComputeFloatParam(cs, s_MediumStepSize, Parameters.mediumStepSize);
            cmd.SetComputeFloatParam(cs, s_Thickness, Parameters.thickness);
            cmd.SetComputeFloatParam(cs, s_ThicknessIncrement, Parameters.thicknessIncrement);
            cmd.SetComputeFloatParam(cs, s_RayCount, Parameters.rayCount);
            cmd.SetComputeFloatParam(cs, s_TemporalIntensity, Parameters.temporalIntensity);
            cmd.SetComputeFloatParam(cs, s_MaxBrightness, Parameters.maxBrightness);
            cmd.SetComputeFloatParam(cs, s_DownSample, Parameters.downSample);
            cmd.SetComputeFloatParam(cs, s_FrameIndex, m_FrameCount);
            cmd.SetComputeFloatParam(cs, s_HistoryTextureValid, historyData.isValid ? 1.0f : 0.0f);
            cmd.SetComputeFloatParam(cs, s_IsProbeCamera, Parameters.isProbeCamera);
            cmd.SetComputeFloatParam(cs, s_BackDepthEnabled, Parameters.backDepthEnabled);
            cmd.SetComputeFloatParam(cs, s_IndirectDiffuseLightingMultiplier, Parameters.indirectDiffuseLightingMultiplier);
            cmd.SetComputeIntParam(cs, s_IndirectDiffuseRenderingLayers, (int)Parameters.indirectDiffuseRenderingLayers);
            cmd.SetComputeFloatParam(cs, s_AggressiveDenoise, Parameters.aggressiveDenoise);
            cmd.SetComputeVectorParam(cs, s_ReBlurBlurRotator, EvaluateRotator(k_BlurRands[m_FrameCount % 32]));
            cmd.SetComputeFloatParam(cs, s_ReBlurDenoiserRadius, Parameters.reBlurDenoiserRadius);

            // Previous frame matrices
            if (historyData.isValid)
            {
                cmd.SetComputeMatrixParam(cs, s_PrevInvViewProjMatrix, historyData.prevInvViewProjMatrix);
                cmd.SetComputeVectorParam(cs, s_PrevCameraPositionWS, historyData.prevCameraPositionWS);
            }
            else
            {
                cmd.SetComputeMatrixParam(cs, s_PrevInvViewProjMatrix, camera.previousViewProjectionMatrix.inverse);
                cmd.SetComputeVectorParam(cs, s_PrevCameraPositionWS, camera.transform.position);
            }

            // Pixel spread angle
            float fov = camera.orthographic ? 1.0f : camera.fieldOfView;
            float pixelSpread = Mathf.Tan(fov * Mathf.Deg2Rad * 0.5f) * 2.0f / Mathf.Min(indirectWidth, indirectHeight);
            cmd.SetComputeFloatParam(cs, s_PixelSpreadAngleTangent, pixelSpread);
        }

        // Set ambient SH if overriding
        if (Parameters.overrideAmbientLighting)
        {
            SphericalHarmonicsL2 probe = RenderSettings.ambientProbe;
            foreach (var cs in shaders)
            {
                cmd.SetComputeVectorParam(cs, s_SSGI_SHAr, new Vector4(probe[0, 3], probe[0, 1], probe[0, 2], probe[0, 0] - probe[0, 6]));
                cmd.SetComputeVectorParam(cs, s_SSGI_SHAg, new Vector4(probe[1, 3], probe[1, 1], probe[1, 2], probe[1, 0] - probe[1, 6]));
                cmd.SetComputeVectorParam(cs, s_SSGI_SHAb, new Vector4(probe[2, 3], probe[2, 1], probe[2, 2], probe[2, 0] - probe[2, 6]));
                cmd.SetComputeVectorParam(cs, s_SSGI_SHBr, new Vector4(probe[0, 4], probe[0, 5], probe[0, 6] * 3, probe[0, 7]));
                cmd.SetComputeVectorParam(cs, s_SSGI_SHBg, new Vector4(probe[1, 4], probe[1, 5], probe[1, 6] * 3, probe[1, 7]));
                cmd.SetComputeVectorParam(cs, s_SSGI_SHBb, new Vector4(probe[2, 4], probe[2, 5], probe[2, 6] * 3, probe[2, 7]));
                cmd.SetComputeVectorParam(cs, s_SSGI_SHC, new Vector4(probe[0, 8], probe[1, 8], probe[2, 8], 1));
            }
        }
    }

    private void DispatchRayMarching(CommandBuffer cmd, int width, int height, CameraHistoryData historyData)
    {
        cmd.SetComputeTextureParam(m_RayMarchingCS, m_RayMarchingKernel, s_SSGIHistoryCameraColorTexture, historyData.historyCameraColorHandle);
        cmd.SetComputeTextureParam(m_RayMarchingCS, m_RayMarchingKernel, s_SSGIHistoryDepthTexture, historyData.historyDepthHandle);
        cmd.SetComputeTextureParam(m_RayMarchingCS, m_RayMarchingKernel, s_SSGIOutput, m_IndirectDiffuseHandle);

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        cmd.DispatchCompute(m_RayMarchingCS, m_RayMarchingKernel, threadGroupsX, threadGroupsY, 1);
    }

    private void DispatchTemporalReprojection(CommandBuffer cmd, int width, int height, CameraHistoryData historyData)
    {
        cmd.SetComputeTextureParam(m_TemporalFilterCS, m_TemporalReprojectionKernel, s_SSGIInputTexture, m_IndirectDiffuseHandle);
        cmd.SetComputeTextureParam(m_TemporalFilterCS, m_TemporalReprojectionKernel, s_HistoryIndirectDiffuseTexture, historyData.historyIndirectDiffuseHandle);
        cmd.SetComputeTextureParam(m_TemporalFilterCS, m_TemporalReprojectionKernel, s_SSGIHistorySampleTexture, historyData.historySampleHandle);
        cmd.SetComputeTextureParam(m_TemporalFilterCS, m_TemporalReprojectionKernel, s_SSGIOutput, m_IntermediateHandle);
        cmd.SetComputeTextureParam(m_TemporalFilterCS, m_TemporalReprojectionKernel, s_SSGISampleOutput, m_SampleCountHandle);

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        cmd.DispatchCompute(m_TemporalFilterCS, m_TemporalReprojectionKernel, threadGroupsX, threadGroupsY, 1);

        // Swap buffers
        CoreUtils.Swap(ref m_IndirectDiffuseHandle, ref m_IntermediateHandle);
    }

    private void DispatchEdgeAwareDenoise(CommandBuffer cmd, int width, int height)
    {
        cmd.SetComputeTextureParam(m_SpatialDenoiseCS, m_EdgeAwareDenoiseKernel, s_SSGIInputTexture, m_IndirectDiffuseHandle);
        cmd.SetComputeTextureParam(m_SpatialDenoiseCS, m_EdgeAwareDenoiseKernel, s_SSGIOutput, m_IntermediateHandle);

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        cmd.DispatchCompute(m_SpatialDenoiseCS, m_EdgeAwareDenoiseKernel, threadGroupsX, threadGroupsY, 1);

        CoreUtils.Swap(ref m_IndirectDiffuseHandle, ref m_IntermediateHandle);
    }

    private void DispatchPoissonDiskDenoise(CommandBuffer cmd, int width, int height)
    {
        cmd.SetComputeTextureParam(m_SpatialDenoiseCS, m_PoissonDiskDenoiseKernel, s_SSGIInputTexture, m_IndirectDiffuseHandle);
        cmd.SetComputeTextureParam(m_SpatialDenoiseCS, m_PoissonDiskDenoiseKernel, s_SSGIOutput, m_IntermediateHandle);

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        cmd.DispatchCompute(m_SpatialDenoiseCS, m_PoissonDiskDenoiseKernel, threadGroupsX, threadGroupsY, 1);

        CoreUtils.Swap(ref m_IndirectDiffuseHandle, ref m_IntermediateHandle);
    }

    private void DispatchTemporalStabilization(CommandBuffer cmd, int width, int height, CameraHistoryData historyData)
    {
        cmd.SetComputeTextureParam(m_TemporalFilterCS, m_TemporalStabilizationKernel, s_SSGIInputTexture, m_IndirectDiffuseHandle);
        cmd.SetComputeTextureParam(m_TemporalFilterCS, m_TemporalStabilizationKernel, s_HistoryIndirectDiffuseTexture, historyData.historyIndirectDiffuseHandle);
        cmd.SetComputeTextureParam(m_TemporalFilterCS, m_TemporalStabilizationKernel, s_SSGIOutput, m_IntermediateHandle);

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        cmd.DispatchCompute(m_TemporalFilterCS, m_TemporalStabilizationKernel, threadGroupsX, threadGroupsY, 1);

        CoreUtils.Swap(ref m_IndirectDiffuseHandle, ref m_IntermediateHandle);
    }

    private void DispatchHistoryUpdate(CommandBuffer cmd, int width, int height, CameraHistoryData historyData)
    {
        cmd.SetComputeTextureParam(m_HistoryUpdateCS, m_UpdateHistoryBuffersKernel, s_IndirectDiffuseTexture, m_IndirectDiffuseHandle);
        cmd.SetComputeTextureParam(m_HistoryUpdateCS, m_UpdateHistoryBuffersKernel, s_SSGISampleTexture, m_SampleCountHandle);
        cmd.SetComputeTextureParam(m_HistoryUpdateCS, m_UpdateHistoryBuffersKernel, s_HistoryDepthOutput, historyData.historyDepthHandle);
        cmd.SetComputeTextureParam(m_HistoryUpdateCS, m_UpdateHistoryBuffersKernel, s_HistoryIndirectOutput, historyData.historyIndirectDiffuseHandle);
        cmd.SetComputeTextureParam(m_HistoryUpdateCS, m_UpdateHistoryBuffersKernel, s_HistorySampleOutput, historyData.historySampleHandle);

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        cmd.DispatchCompute(m_HistoryUpdateCS, m_UpdateHistoryBuffersKernel, threadGroupsX, threadGroupsY, 1);
    }

    private void DispatchCopyHistoryDepth(CommandBuffer cmd, int width, int height, CameraHistoryData historyData)
    {
        cmd.SetComputeTextureParam(m_CombineCS, m_CopyHistoryDepthKernel, s_HistoryDepthOutput, historyData.historyDepthHandle);

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        cmd.DispatchCompute(m_CombineCS, m_CopyHistoryDepthKernel, threadGroupsX, threadGroupsY, 1);
    }

    // FIX: Updated DispatchCombine to use intermediate UAV texture and blit back
    private void DispatchCombine(CommandBuffer cmd, int width, int height, RTHandle colorTarget)
    {
        // Read from color target, write to UAV-compatible intermediate
        cmd.SetComputeTextureParam(m_CombineCS, m_CombineGIKernel, s_IndirectDiffuseTexture, m_IndirectDiffuseHandle);
        cmd.SetComputeTextureParam(m_CombineCS, m_CombineGIKernel, s_SSGIInputTexture, colorTarget);
        cmd.SetComputeTextureParam(m_CombineCS, m_CombineGIKernel, s_SSGIOutput, m_CombineOutputHandle);

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        cmd.DispatchCompute(m_CombineCS, m_CombineGIKernel, threadGroupsX, threadGroupsY, 1);

        // Blit result back to camera color target
        cmd.Blit(m_CombineOutputHandle, colorTarget);
    }

    private void DispatchCopyHistoryColor(CommandBuffer cmd, int width, int height, RTHandle colorTarget, CameraHistoryData historyData)
    {
        cmd.SetComputeTextureParam(m_CombineCS, m_CopyHistoryColorKernel, s_CameraColorTexture, colorTarget);
        cmd.SetComputeTextureParam(m_CombineCS, m_CopyHistoryColorKernel, s_HistoryColorOutput, historyData.historyCameraColorHandle);

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        cmd.DispatchCompute(m_CombineCS, m_CopyHistoryColorKernel, threadGroupsX, threadGroupsY, 1);
    }
    #endregion

#if UNITY_6000_0_OR_NEWER
    #region Render Graph
    // Render Graph implementation would follow similar pattern
    // For brevity, showing the structure
    
    private class MainPassData
    {
        // All the texture handles and parameters needed
        public ComputeShader[] computeShaders;
        public int[] kernels;
        public TextureHandle colorTexture;
        public TextureHandle depthTexture;
        public TextureHandle motionTexture;
        public TextureHandle indirectDiffuse;
        public TextureHandle intermediate;
        public TextureHandle sampleCount;
        public TextureHandle combineOutput;
        // History textures...
        public SSGIParameters parameters;
        public int frameCount;
        public bool historyValid;
    }

    public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
    {
        // Similar implementation to Execute, but using RenderGraph API
        // This would create multiple compute passes for each stage
        
        using (var builder = renderGraph.AddComputePass<MainPassData>(k_ProfilerTag, out var passData))
        {
            UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
            UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();

            // Setup pass data...
            // The actual implementation would be similar to the non-RG version
            // but using TextureHandle instead of RTHandle and proper RG resource management

            builder.SetRenderFunc((MainPassData data, ComputeGraphContext context) =>
            {
                // Execute compute dispatches similar to the non-RG Execute method
            });
        }
    }
    #endregion
#endif

    public void Dispose()
    {
        m_IndirectDiffuseHandle?.Release();
        m_IntermediateHandle?.Release();
        m_SampleCountHandle?.Release();
        m_CombineOutputHandle?.Release(); // NEW: Dispose the combine output handle

        for (int i = 0; i < k_MaxCameraCount; i++)
        {
            m_CameraHistoryData[i].historyDepthHandle?.Release();
            m_CameraHistoryData[i].historyCameraColorHandle?.Release();
            m_CameraHistoryData[i].historyIndirectDiffuseHandle?.Release();
            m_CameraHistoryData[i].historySampleHandle?.Release();
        }
    }
}
