using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

#if UNITY_6000_0_OR_NEWER
using UnityEngine.Rendering.RenderGraphModule;
#endif

public class BackfaceDataPass : ScriptableRenderPass
{
    private const string k_ProfilerTag = "Render Backface Data";
    private readonly ProfilingSampler m_ProfilingSampler = new ProfilingSampler(k_ProfilerTag);

    private RTHandle m_BackDepthHandle;
    private RTHandle m_BackColorHandle;
    
    public bool BackfaceLighting { get; set; }

    private RenderStateBlock m_RenderStateBlock = new RenderStateBlock(RenderStateMask.Nothing);
    private readonly List<ShaderTagId> m_LitTags = new List<ShaderTagId>();

    private static readonly ShaderTagId s_DepthOnly = new ShaderTagId("DepthOnly");
    private static readonly ShaderTagId s_UniversalForward = new ShaderTagId("UniversalForward");
    private static readonly ShaderTagId s_UniversalForwardOnly = new ShaderTagId("UniversalForwardOnly");

    private static readonly int s_CameraBackDepthTexture = Shader.PropertyToID("_CameraBackDepthTexture");
    private static readonly int s_CameraBackOpaqueTexture = Shader.PropertyToID("_CameraBackOpaqueTexture");

    #region Non Render Graph
#if UNITY_6000_0_OR_NEWER
    [System.Obsolete]
#endif
    public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
    {
        var depthDesc = renderingData.cameraData.cameraTargetDescriptor;
        depthDesc.msaaSamples = 1;
        depthDesc.bindMS = false;
        depthDesc.graphicsFormat = GraphicsFormat.None;

        if (!BackfaceLighting)
        {
        #if UNITY_6000_0_OR_NEWER
            RenderingUtils.ReAllocateHandleIfNeeded(ref m_BackDepthHandle, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_CameraBackDepthTexture");
        #else
            RenderingUtils.ReAllocateIfNeeded(ref m_BackDepthHandle, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_CameraBackDepthTexture");
        #endif
            cmd.SetGlobalTexture(s_CameraBackDepthTexture, m_BackDepthHandle);
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

            cmd.SetGlobalTexture(s_CameraBackDepthTexture, m_BackDepthHandle);
            cmd.SetGlobalTexture(s_CameraBackOpaqueTexture, m_BackColorHandle);
            ConfigureTarget(m_BackColorHandle, m_BackDepthHandle);
            ConfigureClear(ClearFlag.Color | ClearFlag.Depth, Color.clear);
        }
    }

#if UNITY_6000_0_OR_NEWER
    [System.Obsolete]
#endif
    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        CommandBuffer cmd = CommandBufferPool.Get();

        using (new ProfilingScope(cmd, m_ProfilingSampler))
        {
            ShaderTagId passTag = BackfaceLighting ? s_UniversalForward : s_DepthOnly;
            
            if (BackfaceLighting)
            {
                m_LitTags.Add(s_UniversalForward);
                m_LitTags.Add(s_UniversalForwardOnly);
            }

            m_RenderStateBlock.depthState = new DepthState(true, CompareFunction.LessEqual);
            m_RenderStateBlock.mask |= RenderStateMask.Depth;
            m_RenderStateBlock.rasterState = new RasterState(CullMode.Front);
            m_RenderStateBlock.mask |= RenderStateMask.Raster;

            var sortingCriteria = renderingData.cameraData.defaultOpaqueSortFlags;
            var drawSettings = CreateDrawingSettings(BackfaceLighting ? m_LitTags : new List<ShaderTagId> { s_DepthOnly },
                ref renderingData, sortingCriteria);
            
            var filterSettings = new FilteringSettings(RenderQueueRange.opaque);

            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();

            context.DrawRenderers(renderingData.cullResults, ref drawSettings, ref filterSettings, ref m_RenderStateBlock);
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
        public RendererListHandle rendererListHandle;
    }

    public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
    {
        using (var builder = renderGraph.AddRasterRenderPass<PassData>(k_ProfilerTag, out var passData))
        {
            UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
            UniversalRenderingData renderingData = frameData.Get<UniversalRenderingData>();
            UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();

            // Create textures
            TextureDesc depthDesc;
            if (!resourceData.isActiveTargetBackBuffer)
                depthDesc = resourceData.activeDepthTexture.GetDescriptor(renderGraph);
            else
            {
                depthDesc = resourceData.cameraDepthTexture.GetDescriptor(renderGraph);
                var backBufferInfo = renderGraph.GetRenderTargetInfo(resourceData.backBufferDepth);
                depthDesc.colorFormat = backBufferInfo.format;
            }
            depthDesc.name = "_CameraBackDepthTexture";
            depthDesc.clearBuffer = true;
            depthDesc.msaaSamples = MSAASamples.None;

            TextureHandle backDepthHandle = renderGraph.CreateTexture(depthDesc);

            if (!BackfaceLighting)
            {
                m_RenderStateBlock.depthState = new DepthState(true, CompareFunction.LessEqual);
                m_RenderStateBlock.mask |= RenderStateMask.Depth;
                m_RenderStateBlock.rasterState = new RasterState(CullMode.Front);
                m_RenderStateBlock.mask |= RenderStateMask.Raster;

                var rendererListDesc = new RendererListDesc(s_DepthOnly, renderingData.cullResults, cameraData.camera)
                {
                    stateBlock = m_RenderStateBlock,
                    sortingCriteria = cameraData.defaultOpaqueSortFlags,
                    renderQueueRange = RenderQueueRange.opaque
                };

                passData.rendererListHandle = renderGraph.CreateRendererList(rendererListDesc);
                builder.UseRendererList(passData.rendererListHandle);
                builder.SetRenderAttachmentDepth(backDepthHandle, AccessFlags.ReadWrite);
                builder.SetGlobalTextureAfterPass(backDepthHandle, s_CameraBackDepthTexture);
            }
            else
            {
                var colorDesc = resourceData.cameraColor.GetDescriptor(renderGraph);
                colorDesc.name = "_CameraBackOpaqueTexture";
                colorDesc.clearBuffer = true;
                colorDesc.msaaSamples = MSAASamples.None;
                colorDesc.colorFormat = GraphicsFormat.B10G11R11_UFloatPack32;

                TextureHandle backColorHandle = renderGraph.CreateTexture(colorDesc);

                m_LitTags[0] = s_UniversalForward;
                m_LitTags[1] = s_UniversalForwardOnly;

                m_RenderStateBlock.depthState = new DepthState(true, CompareFunction.LessEqual);
                m_RenderStateBlock.mask |= RenderStateMask.Depth;
                m_RenderStateBlock.rasterState = new RasterState(CullMode.Front);
                m_RenderStateBlock.mask |= RenderStateMask.Raster;

                var rendererListDesc = new RendererListDesc(m_LitTags, renderingData.cullResults, cameraData.camera)
                {
                    stateBlock = m_RenderStateBlock,
                    sortingCriteria = cameraData.defaultOpaqueSortFlags,
                    renderQueueRange = RenderQueueRange.opaque
                };

                passData.rendererListHandle = renderGraph.CreateRendererList(rendererListDesc);
                builder.UseRendererList(passData.rendererListHandle);
                builder.SetRenderAttachment(backColorHandle, 0);
                builder.SetRenderAttachmentDepth(backDepthHandle);
                builder.SetGlobalTextureAfterPass(backColorHandle, s_CameraBackOpaqueTexture);
                builder.SetGlobalTextureAfterPass(backDepthHandle, s_CameraBackDepthTexture);
            }

            builder.SetRenderFunc((PassData data, RasterGraphContext context) =>
            {
                context.cmd.DrawRendererList(data.rendererListHandle);
            });
        }
    }
    #endregion
#endif

    public void Dispose()
    {
        m_BackDepthHandle?.Release();
        m_BackColorHandle?.Release();
    }
}

public class ForwardGBufferPass : ScriptableRenderPass
{
    private const string k_ProfilerTag = "Render Forward GBuffer";
    private readonly ProfilingSampler m_ProfilingSampler = new ProfilingSampler(k_ProfilerTag);

    private static readonly ShaderTagId s_GBufferPass = new ShaderTagId("UniversalGBuffer");

    private RTHandle m_GBuffer0;
    private RTHandle m_GBuffer1;
    private RTHandle m_GBuffer2;
    private RTHandle m_GBufferDepth;

    private RenderStateBlock m_RenderStateBlock = new RenderStateBlock(RenderStateMask.Nothing);

    private static readonly int s_GBuffer0ID = Shader.PropertyToID("_GBuffer0");
    private static readonly int s_GBuffer1ID = Shader.PropertyToID("_GBuffer1");
    private static readonly int s_GBuffer2ID = Shader.PropertyToID("_GBuffer2");

    private GraphicsFormat GetGBufferFormat(int index)
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
        return GraphicsFormat.None;
    }

    #region Non Render Graph
#if UNITY_6000_0_OR_NEWER
    [System.Obsolete]
#endif
    public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
    {
        var desc = renderingData.cameraData.cameraTargetDescriptor;
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
        cmd.SetGlobalTexture(s_GBuffer0ID, m_GBuffer0);

        desc.graphicsFormat = GetGBufferFormat(1);
    #if UNITY_6000_0_OR_NEWER
        RenderingUtils.ReAllocateHandleIfNeeded(ref m_GBuffer1, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBuffer1");
    #else
        RenderingUtils.ReAllocateIfNeeded(ref m_GBuffer1, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBuffer1");
    #endif
        cmd.SetGlobalTexture(s_GBuffer1ID, m_GBuffer1);

        desc.graphicsFormat = GetGBufferFormat(2);
    #if UNITY_6000_0_OR_NEWER
        RenderingUtils.ReAllocateHandleIfNeeded(ref m_GBuffer2, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBuffer2");
    #else
        RenderingUtils.ReAllocateIfNeeded(ref m_GBuffer2, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBuffer2");
    #endif
        cmd.SetGlobalTexture(s_GBuffer2ID, m_GBuffer2);

        var depthDesc = renderingData.cameraData.cameraTargetDescriptor;
        depthDesc.msaaSamples = 1;
        depthDesc.bindMS = false;
        depthDesc.graphicsFormat = GraphicsFormat.None;

    #if UNITY_6000_0_OR_NEWER
        RenderingUtils.ReAllocateHandleIfNeeded(ref m_GBufferDepth, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBufferDepthTexture");
    #else
        RenderingUtils.ReAllocateIfNeeded(ref m_GBufferDepth, depthDesc, FilterMode.Point, TextureWrapMode.Clamp, name: "_GBufferDepthTexture");
    #endif

        bool isOpenGL = SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLES3 || 
                        SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLCore;
        
        bool canDepthPriming = !isOpenGL && 
            (renderingData.cameraData.renderType == CameraRenderType.Base || renderingData.cameraData.clearDepth) &&
            renderingData.cameraData.cameraTargetDescriptor.msaaSamples == desc.msaaSamples;

        if (canDepthPriming)
        {
            ConfigureTarget(new RTHandle[] { m_GBuffer0, m_GBuffer1, m_GBuffer2 }, 
                renderingData.cameraData.renderer.cameraDepthTargetHandle);
            m_RenderStateBlock.depthState = new DepthState(false, CompareFunction.Equal);
            m_RenderStateBlock.mask |= RenderStateMask.Depth;
        }
        else
        {
            ConfigureTarget(new RTHandle[] { m_GBuffer0, m_GBuffer1, m_GBuffer2 }, m_GBufferDepth);
            m_RenderStateBlock.depthState = new DepthState(true, CompareFunction.LessEqual);
            m_RenderStateBlock.mask |= RenderStateMask.Depth;
        }

        if (isOpenGL)
            ConfigureClear(ClearFlag.Color | ClearFlag.Depth, Color.black);
        else
            ConfigureClear(ClearFlag.Color, Color.clear);

        ConfigureInput(ScriptableRenderPassInput.Depth);
    }

#if UNITY_6000_0_OR_NEWER
    [System.Obsolete]
#endif
    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        var sortingCriteria = renderingData.cameraData.defaultOpaqueSortFlags;

        CommandBuffer cmd = CommandBufferPool.Get();
        using (new ProfilingScope(cmd, m_ProfilingSampler))
        {
            var drawSettings = CreateDrawingSettings(s_GBufferPass, ref renderingData, sortingCriteria);
            var filterSettings = new FilteringSettings(RenderQueueRange.opaque);

            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();

            context.DrawRenderers(renderingData.cullResults, ref drawSettings, ref filterSettings, ref m_RenderStateBlock);
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
        public bool isOpenGL;
        public RendererListHandle rendererListHandle;
    }

    public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
    {
        using (var builder = renderGraph.AddRasterRenderPass<PassData>(k_ProfilerTag, out var passData))
        {
            UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
            UniversalRenderingData renderingData = frameData.Get<UniversalRenderingData>();
            UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();

            var desc = cameraData.cameraTargetDescriptor;
            desc.msaaSamples = 1;
            desc.bindMS = false;
            desc.depthBufferBits = 0;

            desc.graphicsFormat = GetGBufferFormat(0);
            TextureHandle gBuffer0 = UniversalRenderer.CreateRenderGraphTexture(renderGraph, desc, "_GBuffer0", false);

            desc.graphicsFormat = GetGBufferFormat(1);
            TextureHandle gBuffer1 = UniversalRenderer.CreateRenderGraphTexture(renderGraph, desc, "_GBuffer1", false);

            desc.graphicsFormat = GetGBufferFormat(2);
            TextureHandle gBuffer2 = UniversalRenderer.CreateRenderGraphTexture(renderGraph, desc, "_GBuffer2", false);

            bool isOpenGL = SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLES3 ||
                            SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLCore;

            bool canDepthPriming = !isOpenGL &&
                (cameraData.renderType == CameraRenderType.Base || cameraData.clearDepth) &&
                cameraData.cameraTargetDescriptor.msaaSamples == 1;

            TextureHandle depthHandle;
            if (canDepthPriming)
            {
                depthHandle = resourceData.activeDepthTexture;
                m_RenderStateBlock.depthState = new DepthState(false, CompareFunction.Equal);
                m_RenderStateBlock.mask |= RenderStateMask.Depth;
            }
            else
            {
                TextureDesc depthDesc;
                if (!resourceData.isActiveTargetBackBuffer)
                    depthDesc = resourceData.activeDepthTexture.GetDescriptor(renderGraph);
                else
                {
                    depthDesc = resourceData.cameraDepthTexture.GetDescriptor(renderGraph);
                    var backBufferInfo = renderGraph.GetRenderTargetInfo(resourceData.backBufferDepth);
                    depthDesc.colorFormat = backBufferInfo.format;
                }
                depthDesc.name = "_GBufferDepthTexture";
                depthDesc.clearBuffer = false;
                depthDesc.msaaSamples = MSAASamples.None;

                depthHandle = renderGraph.CreateTexture(depthDesc);
                m_RenderStateBlock.depthState = new DepthState(true, CompareFunction.LessEqual);
                m_RenderStateBlock.mask |= RenderStateMask.Depth;
            }

            passData.isOpenGL = isOpenGL;

            var rendererListDesc = new RendererListDesc(s_GBufferPass, renderingData.cullResults, cameraData.camera)
            {
                stateBlock = m_RenderStateBlock,
                sortingCriteria = cameraData.defaultOpaqueSortFlags,
                renderQueueRange = RenderQueueRange.opaque
            };

            passData.rendererListHandle = renderGraph.CreateRendererList(rendererListDesc);
            builder.UseRendererList(passData.rendererListHandle);

            builder.SetRenderAttachment(gBuffer0, 0);
            builder.SetRenderAttachment(gBuffer1, 1);
            builder.SetRenderAttachment(gBuffer2, 2);
            builder.SetRenderAttachmentDepth(depthHandle, AccessFlags.Write);

            builder.SetGlobalTextureAfterPass(gBuffer0, s_GBuffer0ID);
            builder.SetGlobalTextureAfterPass(gBuffer1, s_GBuffer1ID);
            builder.SetGlobalTextureAfterPass(gBuffer2, s_GBuffer2ID);

            builder.SetRenderFunc((PassData data, RasterGraphContext context) =>
            {
                if (data.isOpenGL)
                    context.cmd.ClearRenderTarget(true, true, Color.black);

                context.cmd.DrawRendererList(data.rendererListHandle);
            });
        }
    }
    #endregion
#endif

    public void Dispose()
    {
        m_GBuffer0?.Release();
        m_GBuffer1?.Release();
        m_GBuffer2?.Release();
        m_GBufferDepth?.Release();
    }
}
