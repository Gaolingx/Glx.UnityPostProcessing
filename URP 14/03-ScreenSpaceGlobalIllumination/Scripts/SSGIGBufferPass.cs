using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Experimental.Rendering;

#if UNITY_6000_0_OR_NEWER
using UnityEngine.Rendering.RenderGraphModule;
#endif

public class SSGIGBufferPass : ScriptableRenderPass
{
    private const string k_ProfilerTag = "SSGI GBuffer Processing";
    private readonly ProfilingSampler m_ProfilingSampler = new ProfilingSampler(k_ProfilerTag);

    private readonly ComputeShader m_GBufferCS;
    private int m_CopyDirectLightingKernel;

    private RTHandle m_DirectLightingHandle;
    private RTHandle m_APVLightingHandle;

    public SSGIParameters Parameters { get; set; }

    // Shader property IDs
    private static readonly int s_CameraColorTexture = Shader.PropertyToID("_CameraColorTexture");
    private static readonly int s_CameraDepthTexture = Shader.PropertyToID("_CameraDepthTexture");
    private static readonly int s_GBuffer0 = Shader.PropertyToID("_GBuffer0");
    private static readonly int s_GBuffer1 = Shader.PropertyToID("_GBuffer1");
    private static readonly int s_GBuffer2 = Shader.PropertyToID("_GBuffer2");
    private static readonly int s_DirectLightingOutput = Shader.PropertyToID("_DirectLightingOutput");
    private static readonly int s_APVLightingOutput = Shader.PropertyToID("_APVLightingOutput");
    private static readonly int s_SSGITextureSizes = Shader.PropertyToID("_SSGITextureSizes");
    private static readonly int s_DirectLightingTexture = Shader.PropertyToID("_DirectLightingTexture");
    private static readonly int s_APVLightingTexture = Shader.PropertyToID("_APVLightingTexture");

    // SH coefficients
    private static readonly int s_SSGI_SHAr = Shader.PropertyToID("_SSGI_SHAr");
    private static readonly int s_SSGI_SHAg = Shader.PropertyToID("_SSGI_SHAg");
    private static readonly int s_SSGI_SHAb = Shader.PropertyToID("_SSGI_SHAb");
    private static readonly int s_SSGI_SHBr = Shader.PropertyToID("_SSGI_SHBr");
    private static readonly int s_SSGI_SHBg = Shader.PropertyToID("_SSGI_SHBg");
    private static readonly int s_SSGI_SHBb = Shader.PropertyToID("_SSGI_SHBb");
    private static readonly int s_SSGI_SHC = Shader.PropertyToID("_SSGI_SHC");

    public SSGIGBufferPass(ComputeShader computeShader)
    {
        m_GBufferCS = computeShader;
        m_CopyDirectLightingKernel = m_GBufferCS.FindKernel("CSCopyDirectLighting");
    }

    #region Non Render Graph
#if UNITY_6000_0_OR_NEWER
    [System.Obsolete]
#endif
    public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
    {
        var desc = renderingData.cameraData.cameraTargetDescriptor;
        desc.depthBufferBits = 0;
        desc.msaaSamples = 1;
        desc.graphicsFormat = GraphicsFormat.B10G11R11_UFloatPack32;
        desc.enableRandomWrite = true;

    #if UNITY_6000_0_OR_NEWER
        RenderingUtils.ReAllocateHandleIfNeeded(ref m_DirectLightingHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_DirectLightingTexture");
        RenderingUtils.ReAllocateHandleIfNeeded(ref m_APVLightingHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_APVLightingTexture");
    #else
        RenderingUtils.ReAllocateIfNeeded(ref m_DirectLightingHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_DirectLightingTexture");
        RenderingUtils.ReAllocateIfNeeded(ref m_APVLightingHandle, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_APVLightingTexture");
    #endif

        ConfigureInput(ScriptableRenderPassInput.Depth);
    }

#if UNITY_6000_0_OR_NEWER
    [System.Obsolete]
#endif
    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        if (!Parameters.overrideAmbientLighting)
            return;

        CommandBuffer cmd = CommandBufferPool.Get();

        using (new ProfilingScope(cmd, m_ProfilingSampler))
        {
            var desc = renderingData.cameraData.cameraTargetDescriptor;

            // Set ambient SH coefficients
            SetAmbientSHCoefficients(cmd);

            // Setup compute shader parameters
            cmd.SetComputeVectorParam(m_GBufferCS, s_SSGITextureSizes, 
                new Vector4(desc.width, desc.height, 1.0f / desc.width, 1.0f / desc.height));

            cmd.SetComputeTextureParam(m_GBufferCS, m_CopyDirectLightingKernel, s_CameraColorTexture,
                renderingData.cameraData.renderer.cameraColorTargetHandle);
            cmd.SetComputeTextureParam(m_GBufferCS, m_CopyDirectLightingKernel, s_CameraDepthTexture,
                renderingData.cameraData.renderer.cameraDepthTargetHandle);
            cmd.SetComputeTextureParam(m_GBufferCS, m_CopyDirectLightingKernel, s_DirectLightingOutput, m_DirectLightingHandle);
            cmd.SetComputeTextureParam(m_GBufferCS, m_CopyDirectLightingKernel, s_APVLightingOutput, m_APVLightingHandle);

            int threadGroupsX = Mathf.CeilToInt(desc.width / 8.0f);
            int threadGroupsY = Mathf.CeilToInt(desc.height / 8.0f);
            cmd.DispatchCompute(m_GBufferCS, m_CopyDirectLightingKernel, threadGroupsX, threadGroupsY, 1);

            // Set global textures
            cmd.SetGlobalTexture(s_DirectLightingTexture, m_DirectLightingHandle);
            cmd.SetGlobalTexture(s_APVLightingTexture, m_APVLightingHandle);
        }

        context.ExecuteCommandBuffer(cmd);
        cmd.Clear();
        CommandBufferPool.Release(cmd);
    }

    private void SetAmbientSHCoefficients(CommandBuffer cmd)
    {
        SphericalHarmonicsL2 ambientProbe = RenderSettings.ambientProbe;

        cmd.SetComputeVectorParam(m_GBufferCS, s_SSGI_SHAr, 
            new Vector4(ambientProbe[0, 3], ambientProbe[0, 1], ambientProbe[0, 2], ambientProbe[0, 0] - ambientProbe[0, 6]));
        cmd.SetComputeVectorParam(m_GBufferCS, s_SSGI_SHAg,
            new Vector4(ambientProbe[1, 3], ambientProbe[1, 1], ambientProbe[1, 2], ambientProbe[1, 0] - ambientProbe[1, 6]));
        cmd.SetComputeVectorParam(m_GBufferCS, s_SSGI_SHAb,
            new Vector4(ambientProbe[2, 3], ambientProbe[2, 1], ambientProbe[2, 2], ambientProbe[2, 0] - ambientProbe[2, 6]));
        cmd.SetComputeVectorParam(m_GBufferCS, s_SSGI_SHBr,
            new Vector4(ambientProbe[0, 4], ambientProbe[0, 5], ambientProbe[0, 6] * 3, ambientProbe[0, 7]));
        cmd.SetComputeVectorParam(m_GBufferCS, s_SSGI_SHBg,
            new Vector4(ambientProbe[1, 4], ambientProbe[1, 5], ambientProbe[1, 6] * 3, ambientProbe[1, 7]));
        cmd.SetComputeVectorParam(m_GBufferCS, s_SSGI_SHBb,
            new Vector4(ambientProbe[2, 4], ambientProbe[2, 5], ambientProbe[2, 6] * 3, ambientProbe[2, 7]));
        cmd.SetComputeVectorParam(m_GBufferCS, s_SSGI_SHC,
            new Vector4(ambientProbe[0, 8], ambientProbe[1, 8], ambientProbe[2, 8], 1));
    }
    #endregion

#if UNITY_6000_0_OR_NEWER
    #region Render Graph
    private class PassData
    {
        public ComputeShader computeShader;
        public int kernel;
        public TextureHandle colorTexture;
        public TextureHandle depthTexture;
        public TextureHandle directLightingOutput;
        public TextureHandle apvLightingOutput;
        public Vector4 textureSizes;
        public SphericalHarmonicsL2 ambientProbe;
    }

    public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
    {
        if (!Parameters.overrideAmbientLighting)
            return;

        using (var builder = renderGraph.AddComputePass<PassData>(k_ProfilerTag, out var passData))
        {
            UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
            UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();

            var desc = cameraData.cameraTargetDescriptor;

            // Create output textures
            var outputDesc = new TextureDesc(desc.width, desc.height)
            {
                colorFormat = GraphicsFormat.B10G11R11_UFloatPack32,
                enableRandomWrite = true,
                name = "DirectLighting"
            };

            passData.computeShader = m_GBufferCS;
            passData.kernel = m_CopyDirectLightingKernel;
            passData.colorTexture = resourceData.activeColorTexture;
            passData.depthTexture = resourceData.cameraDepthTexture;
            passData.directLightingOutput = renderGraph.CreateTexture(outputDesc);
            
            outputDesc.name = "APVLighting";
            passData.apvLightingOutput = renderGraph.CreateTexture(outputDesc);
            
            passData.textureSizes = new Vector4(desc.width, desc.height, 1.0f / desc.width, 1.0f / desc.height);
            passData.ambientProbe = RenderSettings.ambientProbe;

            builder.UseTexture(passData.colorTexture, AccessFlags.Read);
            builder.UseTexture(passData.depthTexture, AccessFlags.Read);
            builder.UseTexture(passData.directLightingOutput, AccessFlags.Write);
            builder.UseTexture(passData.apvLightingOutput, AccessFlags.Write);

            builder.SetRenderFunc((PassData data, ComputeGraphContext context) =>
            {
                var cmd = context.cmd;
                var probe = data.ambientProbe;

                // Set SH coefficients
                cmd.SetComputeVectorParam(data.computeShader, s_SSGI_SHAr,
                    new Vector4(probe[0, 3], probe[0, 1], probe[0, 2], probe[0, 0] - probe[0, 6]));
                cmd.SetComputeVectorParam(data.computeShader, s_SSGI_SHAg,
                    new Vector4(probe[1, 3], probe[1, 1], probe[1, 2], probe[1, 0] - probe[1, 6]));
                cmd.SetComputeVectorParam(data.computeShader, s_SSGI_SHAb,
                    new Vector4(probe[2, 3], probe[2, 1], probe[2, 2], probe[2, 0] - probe[2, 6]));
                cmd.SetComputeVectorParam(data.computeShader, s_SSGI_SHBr,
                    new Vector4(probe[0, 4], probe[0, 5], probe[0, 6] * 3, probe[0, 7]));
                cmd.SetComputeVectorParam(data.computeShader, s_SSGI_SHBg,
                    new Vector4(probe[1, 4], probe[1, 5], probe[1, 6] * 3, probe[1, 7]));
                cmd.SetComputeVectorParam(data.computeShader, s_SSGI_SHBb,
                    new Vector4(probe[2, 4], probe[2, 5], probe[2, 6] * 3, probe[2, 7]));
                cmd.SetComputeVectorParam(data.computeShader, s_SSGI_SHC,
                    new Vector4(probe[0, 8], probe[1, 8], probe[2, 8], 1));

                cmd.SetComputeVectorParam(data.computeShader, s_SSGITextureSizes, data.textureSizes);
                cmd.SetComputeTextureParam(data.computeShader, data.kernel, s_CameraColorTexture, data.colorTexture);
                cmd.SetComputeTextureParam(data.computeShader, data.kernel, s_CameraDepthTexture, data.depthTexture);
                cmd.SetComputeTextureParam(data.computeShader, data.kernel, s_DirectLightingOutput, data.directLightingOutput);
                cmd.SetComputeTextureParam(data.computeShader, data.kernel, s_APVLightingOutput, data.apvLightingOutput);

                int threadGroupsX = Mathf.CeilToInt(data.textureSizes.x / 8.0f);
                int threadGroupsY = Mathf.CeilToInt(data.textureSizes.y / 8.0f);
                cmd.DispatchCompute(data.computeShader, data.kernel, threadGroupsX, threadGroupsY, 1);
            });

            builder.SetGlobalTextureAfterPass(passData.directLightingOutput, s_DirectLightingTexture);
            builder.SetGlobalTextureAfterPass(passData.apvLightingOutput, s_APVLightingTexture);
        }
    }
    #endregion
#endif

    public void Dispose()
    {
        m_DirectLightingHandle?.Release();
        m_APVLightingHandle?.Release();
    }
}
