using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Experimental.Rendering;

#if UNITY_6000_0_OR_NEWER
using UnityEngine.Rendering.RenderGraphModule;
#endif

public class DepthPyramidPass : ScriptableRenderPass
{
    private const string k_ProfilerTag = "SSGI Depth Pyramid";
    private readonly ProfilingSampler m_ProfilingSampler = new ProfilingSampler(k_ProfilerTag);

    private readonly ComputeShader m_DepthPyramidCS;
    private int m_DownsampleFirstKernel;
    private int m_DownsampleKernel;

    private RTHandle m_HZBTexture;
    private RTHandle[] m_HZBMips;
    private const int k_MaxMipCount = 10;

    // Shader property IDs
    private static readonly int s_DepthSource = Shader.PropertyToID("_DepthSource");
    private static readonly int s_DepthMip0 = Shader.PropertyToID("_DepthMip0");
    private static readonly int s_DepthMipInput = Shader.PropertyToID("_DepthMipInput");
    private static readonly int s_DepthMipOutput = Shader.PropertyToID("_DepthMipOutput");
    private static readonly int s_OutputSize = Shader.PropertyToID("_OutputSize");
    private static readonly int s_InputSize = Shader.PropertyToID("_InputSize");
    private static readonly int s_MipLevel = Shader.PropertyToID("_MipLevel");
    private static readonly int s_HZBTexture = Shader.PropertyToID("_HZBTexture");
    private static readonly int s_HZBMipCount = Shader.PropertyToID("_HZBMipCount");
    private static readonly int s_HZBResolution = Shader.PropertyToID("_HZBResolution");
    
    // FIX: Use the global depth texture property
    private static readonly int s_CameraDepthTexture = Shader.PropertyToID("_CameraDepthTexture");

    public DepthPyramidPass(ComputeShader computeShader)
    {
        m_DepthPyramidCS = computeShader;
        m_DownsampleFirstKernel = m_DepthPyramidCS.FindKernel("CSDepthDownsampleFirst");
        m_DownsampleKernel = m_DepthPyramidCS.FindKernel("CSDepthDownsample");
        m_HZBMips = new RTHandle[k_MaxMipCount];
    }

    #region Non Render Graph
#if UNITY_6000_0_OR_NEWER
    [System.Obsolete]
#endif
    public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
    {
        var desc = renderingData.cameraData.cameraTargetDescriptor;
        int width = Mathf.Max(1, desc.width / 2);
        int height = Mathf.Max(1, desc.height / 2);

        // Calculate mip count
        int mipCount = Mathf.Min(k_MaxMipCount, Mathf.FloorToInt(Mathf.Log(Mathf.Max(width, height), 2)) + 1);

        // Allocate HZB mips
        for (int i = 0; i < mipCount; i++)
        {
            int mipWidth = Mathf.Max(1, width >> i);
            int mipHeight = Mathf.Max(1, height >> i);

            var mipDesc = new RenderTextureDescriptor(mipWidth, mipHeight, RenderTextureFormat.RFloat, 0)
            {
                enableRandomWrite = true,
                msaaSamples = 1
            };

        #if UNITY_6000_0_OR_NEWER
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_HZBMips[i], mipDesc, FilterMode.Point, TextureWrapMode.Clamp, name: $"_HZBMip{i}");
        #else
            RenderingUtils.ReAllocateIfNeeded(ref m_HZBMips[i], mipDesc, FilterMode.Point, TextureWrapMode.Clamp, name: $"_HZBMip{i}");
        #endif
        }

        // Set global HZB texture (mip 0 for now, will be updated)
        cmd.SetGlobalTexture(s_HZBTexture, m_HZBMips[0]);
        cmd.SetGlobalInt(s_HZBMipCount, mipCount);
        cmd.SetGlobalVector(s_HZBResolution, new Vector2(width, height));

        // FIX: Request depth input to ensure it's available
        ConfigureInput(ScriptableRenderPassInput.Depth);
    }

#if UNITY_6000_0_OR_NEWER
    [System.Obsolete]
#endif
    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        CommandBuffer cmd = CommandBufferPool.Get();

        using (new ProfilingScope(cmd, m_ProfilingSampler))
        {
            var desc = renderingData.cameraData.cameraTargetDescriptor;
            int width = Mathf.Max(1, desc.width / 2);
            int height = Mathf.Max(1, desc.height / 2);
            int mipCount = Mathf.Min(k_MaxMipCount, Mathf.FloorToInt(Mathf.Log(Mathf.Max(width, height), 2)) + 1);

            // FIX: Get the depth texture handle properly
            // The cameraDepthTargetHandle might be the actual depth buffer or a copy
            // Using the renderer's depth target is more reliable
            RTHandle depthSource = renderingData.cameraData.renderer.cameraDepthTargetHandle;
            
            // Safety check
            if (depthSource == null || depthSource.rt == null)
            {
                Debug.LogWarning("SSGI: Depth texture is not available for HZB generation.");
                context.ExecuteCommandBuffer(cmd);
                cmd.Clear();
                CommandBufferPool.Release(cmd);
                return;
            }

            // First pass: depth buffer to mip 0
            cmd.SetComputeTextureParam(m_DepthPyramidCS, m_DownsampleFirstKernel, s_DepthSource, depthSource);
            cmd.SetComputeTextureParam(m_DepthPyramidCS, m_DownsampleFirstKernel, s_DepthMip0, m_HZBMips[0]);
            cmd.SetComputeVectorParam(m_DepthPyramidCS, s_OutputSize, new Vector2(width, height));
            cmd.SetComputeVectorParam(m_DepthPyramidCS, s_InputSize, new Vector2(desc.width, desc.height));

            int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
            int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
            cmd.DispatchCompute(m_DepthPyramidCS, m_DownsampleFirstKernel, threadGroupsX, threadGroupsY, 1);

            // Subsequent passes: mip N-1 to mip N
            for (int i = 1; i < mipCount; i++)
            {
                int prevWidth = Mathf.Max(1, width >> (i - 1));
                int prevHeight = Mathf.Max(1, height >> (i - 1));
                int currWidth = Mathf.Max(1, width >> i);
                int currHeight = Mathf.Max(1, height >> i);

                cmd.SetComputeTextureParam(m_DepthPyramidCS, m_DownsampleKernel, s_DepthMipInput, m_HZBMips[i - 1]);
                cmd.SetComputeTextureParam(m_DepthPyramidCS, m_DownsampleKernel, s_DepthMipOutput, m_HZBMips[i]);
                cmd.SetComputeVectorParam(m_DepthPyramidCS, s_OutputSize, new Vector2(currWidth, currHeight));
                cmd.SetComputeVectorParam(m_DepthPyramidCS, s_InputSize, new Vector2(prevWidth, prevHeight));
                cmd.SetComputeIntParam(m_DepthPyramidCS, s_MipLevel, i);

                threadGroupsX = Mathf.CeilToInt(currWidth / 8.0f);
                threadGroupsY = Mathf.CeilToInt(currHeight / 8.0f);
                cmd.DispatchCompute(m_DepthPyramidCS, m_DownsampleKernel, threadGroupsX, threadGroupsY, 1);
            }

            // Set global texture to mip 0 (compute shaders will sample with explicit mip levels)
            cmd.SetGlobalTexture(s_HZBTexture, m_HZBMips[0]);
        }

        context.ExecuteCommandBuffer(cmd);
        cmd.Clear();
        CommandBufferPool.Release(cmd);
    }
    #endregion

#if UNITY_6000_0_OR_NEWER
    #region Render Graph
    private class PassData
    {
        public ComputeShader computeShader;
        public int downsampleFirstKernel;
        public int downsampleKernel;
        public TextureHandle depthTexture;
        public TextureHandle[] hzbMips;
        public int mipCount;
        public Vector2Int fullResolution;
        public Vector2Int hzbResolution;
    }

    public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
    {
        using (var builder = renderGraph.AddComputePass<PassData>(k_ProfilerTag, out var passData))
        {
            UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
            UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();

            var desc = cameraData.cameraTargetDescriptor;
            int width = Mathf.Max(1, desc.width / 2);
            int height = Mathf.Max(1, desc.height / 2);
            int mipCount = Mathf.Min(k_MaxMipCount, Mathf.FloorToInt(Mathf.Log(Mathf.Max(width, height), 2)) + 1);

            passData.computeShader = m_DepthPyramidCS;
            passData.downsampleFirstKernel = m_DownsampleFirstKernel;
            passData.downsampleKernel = m_DownsampleKernel;
            passData.depthTexture = resourceData.cameraDepthTexture;
            passData.mipCount = mipCount;
            passData.fullResolution = new Vector2Int(desc.width, desc.height);
            passData.hzbResolution = new Vector2Int(width, height);

            // Create HZB textures
            passData.hzbMips = new TextureHandle[mipCount];
            for (int i = 0; i < mipCount; i++)
            {
                int mipWidth = Mathf.Max(1, width >> i);
                int mipHeight = Mathf.Max(1, height >> i);

                var mipDesc = new TextureDesc(mipWidth, mipHeight)
                {
                    colorFormat = GraphicsFormat.R32_SFloat,
                    enableRandomWrite = true,
                    name = $"HZBMip{i}"
                };

                passData.hzbMips[i] = renderGraph.CreateTexture(mipDesc);
                builder.UseTexture(passData.hzbMips[i], AccessFlags.ReadWrite);
            }

            builder.UseTexture(passData.depthTexture, AccessFlags.Read);

            builder.SetRenderFunc((PassData data, ComputeGraphContext context) =>
            {
                var cmd = context.cmd;
                int width = data.hzbResolution.x;
                int height = data.hzbResolution.y;

                // First pass
                cmd.SetComputeTextureParam(data.computeShader, data.downsampleFirstKernel, s_DepthSource, data.depthTexture);
                cmd.SetComputeTextureParam(data.computeShader, data.downsampleFirstKernel, s_DepthMip0, data.hzbMips[0]);
                cmd.SetComputeVectorParam(data.computeShader, s_OutputSize, new Vector2(width, height));
                cmd.SetComputeVectorParam(data.computeShader, s_InputSize, new Vector2(data.fullResolution.x, data.fullResolution.y));

                int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
                int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
                cmd.DispatchCompute(data.computeShader, data.downsampleFirstKernel, threadGroupsX, threadGroupsY, 1);

                // Subsequent passes
                for (int i = 1; i < data.mipCount; i++)
                {
                    int currWidth = Mathf.Max(1, width >> i);
                    int currHeight = Mathf.Max(1, height >> i);
                    int prevWidth = Mathf.Max(1, width >> (i - 1));
                    int prevHeight = Mathf.Max(1, height >> (i - 1));

                    cmd.SetComputeTextureParam(data.computeShader, data.downsampleKernel, s_DepthMipInput, data.hzbMips[i - 1]);
                    cmd.SetComputeTextureParam(data.computeShader, data.downsampleKernel, s_DepthMipOutput, data.hzbMips[i]);
                    cmd.SetComputeVectorParam(data.computeShader, s_OutputSize, new Vector2(currWidth, currHeight));
                    cmd.SetComputeVectorParam(data.computeShader, s_InputSize, new Vector2(prevWidth, prevHeight));
                    cmd.SetComputeIntParam(data.computeShader, s_MipLevel, i);

                    threadGroupsX = Mathf.CeilToInt(currWidth / 8.0f);
                    threadGroupsY = Mathf.CeilToInt(currHeight / 8.0f);
                    cmd.DispatchCompute(data.computeShader, data.downsampleKernel, threadGroupsX, threadGroupsY, 1);
                }

                // Set globals
                cmd.SetGlobalTexture(s_HZBTexture, data.hzbMips[0]);
                cmd.SetGlobalInt(s_HZBMipCount, data.mipCount);
                cmd.SetGlobalVector(s_HZBResolution, new Vector2(width, height));
            });
        }
    }
    #endregion
#endif

    public void Dispose()
    {
        for (int i = 0; i < k_MaxMipCount; i++)
        {
            m_HZBMips[i]?.Release();
            m_HZBMips[i] = null;
        }
    }
}
