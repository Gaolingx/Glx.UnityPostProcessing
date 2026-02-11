// BokehDOFRenderFeature.cs
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace Glx.PostProcess.URP.Runtime
{
    public class BokehDOFRenderFeature : ScriptableRendererFeature
    {
        [System.Serializable]
        public class Settings
        {
            public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;
            public ComputeShader computeShader;

            [Range(0.25f, 1f)]
            public float downsampleScale = 0.5f;
        }

        public Settings settings = new Settings();
        private BokehDOFRenderPass m_RenderPass;

        public override void Create()
        {
            m_RenderPass = new BokehDOFRenderPass(settings);
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (settings.computeShader == null)
            {
                Debug.LogWarning("BokehDOF: Compute Shader is not assigned!");
                return;
            }

            var cameraType = renderingData.cameraData.cameraType;
            if (cameraType == CameraType.Preview || cameraType == CameraType.Reflection)
                return;

            var volume = VolumeManager.instance.stack.GetComponent<BokehDOFVolume>();
            if (volume == null || !volume.IsActive())
                return;

            m_RenderPass.Setup(settings, volume);
            renderer.EnqueuePass(m_RenderPass);
        }

        protected override void Dispose(bool disposing)
        {
            m_RenderPass?.Dispose();
        }

        //==========================================================================
        // Render Pass
        //==========================================================================
        class BokehDOFRenderPass : ScriptableRenderPass
        {
            private Settings m_Settings;
            private BokehDOFVolume m_Volume;
            private ComputeShader m_ComputeShader;

            // Kernel IDs
            private int m_KernelCoC;
            private int m_KernelDownsample;
            private int m_KernelBokehBlur5Tap;
            private int m_KernelBokehBlur13Tap;
            private int m_KernelComposite;

            // RT Handles
            private RTHandle m_CoCTexture;
            private RTHandle m_DownsampledColor;
            private RTHandle m_DownsampledCoC;
            private RTHandle m_BokehTexture;
            private RTHandle m_ResultTexture;

            // Shader Property IDs
            private static readonly int _SourceTex = Shader.PropertyToID("_SourceTex");
            private static readonly int _DepthTex = Shader.PropertyToID("_DepthTex");
            private static readonly int _CoCTexRead = Shader.PropertyToID("_CoCTexRead");
            private static readonly int _DownsampledColorTexRead = Shader.PropertyToID("_DownsampledColorTexRead");
            private static readonly int _DownsampledCoCTexRead = Shader.PropertyToID("_DownsampledCoCTexRead");
            private static readonly int _BokehTexRead = Shader.PropertyToID("_BokehTexRead");

            // Shader Property IDs
            private static readonly int _CoCTex = Shader.PropertyToID("_CoCTex");
            private static readonly int _DownsampledColorTex = Shader.PropertyToID("_DownsampledColorTex");
            private static readonly int _DownsampledCoCTex = Shader.PropertyToID("_DownsampledCoCTex");
            private static readonly int _BokehTex = Shader.PropertyToID("_BokehTex");
            private static readonly int _ResultTex = Shader.PropertyToID("_ResultTex");

            // Parameter IDs
            private static readonly int _SourceSize = Shader.PropertyToID("_SourceSize");
            private static readonly int _DownsampledSize = Shader.PropertyToID("_DownsampledSize");
            private static readonly int _DOFParams = Shader.PropertyToID("_DOFParams");
            private static readonly int _BlurParams = Shader.PropertyToID("_BlurParams");
            private static readonly int _BokehParams = Shader.PropertyToID("_BokehParams");

            private const string PROFILER_TAG = "Bokeh DOF";
            private ProfilingSampler m_ProfilingSampler = new ProfilingSampler(PROFILER_TAG);

            public BokehDOFRenderPass(Settings settings)
            {
                m_Settings = settings;
                renderPassEvent = settings.renderPassEvent;
            }

            public void Setup(Settings settings, BokehDOFVolume volume)
            {
                m_Settings = settings;
                m_Volume = volume;
                m_ComputeShader = settings.computeShader;

                // Get kernel indices
                m_KernelCoC = m_ComputeShader.FindKernel("CSCalculateCoC");
                m_KernelDownsample = m_ComputeShader.FindKernel("CSDownsample");
                m_KernelBokehBlur5Tap = m_ComputeShader.FindKernel("CSBokehBlur5Tap");
                m_KernelBokehBlur13Tap = m_ComputeShader.FindKernel("CSBokehBlur13Tap");
                m_KernelComposite = m_ComputeShader.FindKernel("CSComposite");
            }

            public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
            {
                var descriptor = renderingData.cameraData.cameraTargetDescriptor;
                descriptor.depthBufferBits = 0;
                descriptor.msaaSamples = 1;

                int width = descriptor.width;
                int height = descriptor.height;
                int downWidth = Mathf.Max(1, Mathf.CeilToInt(width * m_Settings.downsampleScale));
                int downHeight = Mathf.Max(1, Mathf.CeilToInt(height * m_Settings.downsampleScale));

                // Full resolution CoC
                var cocDesc = descriptor;
                cocDesc.colorFormat = RenderTextureFormat.RHalf;
                cocDesc.enableRandomWrite = true;
                RenderingUtils.ReAllocateIfNeeded(ref m_CoCTexture, cocDesc, FilterMode.Bilinear,
                    TextureWrapMode.Clamp, name: "_CoCTexture");

                // Downsampled textures
                var downDesc = descriptor;
                downDesc.width = downWidth;
                downDesc.height = downHeight;
                downDesc.colorFormat = RenderTextureFormat.ARGBHalf;
                downDesc.enableRandomWrite = true;

                RenderingUtils.ReAllocateIfNeeded(ref m_DownsampledColor, downDesc, FilterMode.Bilinear,
                    TextureWrapMode.Clamp, name: "_DownsampledColor");

                var downCoCDesc = downDesc;
                downCoCDesc.colorFormat = RenderTextureFormat.RHalf;
                RenderingUtils.ReAllocateIfNeeded(ref m_DownsampledCoC, downCoCDesc, FilterMode.Bilinear,
                    TextureWrapMode.Clamp, name: "_DownsampledCoC");

                RenderingUtils.ReAllocateIfNeeded(ref m_BokehTexture, downDesc, FilterMode.Bilinear,
                    TextureWrapMode.Clamp, name: "_BokehTexture");

                // Result texture (full resolution)
                var resultDesc = descriptor;
                resultDesc.colorFormat = RenderTextureFormat.ARGBHalf;
                resultDesc.enableRandomWrite = true;
                RenderingUtils.ReAllocateIfNeeded(ref m_ResultTexture, resultDesc, FilterMode.Bilinear,
                    TextureWrapMode.Clamp, name: "_ResultTexture");

                ConfigureInput(ScriptableRenderPassInput.Color | ScriptableRenderPassInput.Depth);
            }

            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                if (m_ComputeShader == null || m_Volume == null)
                    return;

                var cmd = CommandBufferPool.Get(PROFILER_TAG);

                using (new ProfilingScope(cmd, m_ProfilingSampler))
                {
                    ExecuteBokehDOF(cmd, ref renderingData);
                }

                context.ExecuteCommandBuffer(cmd);
                CommandBufferPool.Release(cmd);
            }

            private void ExecuteBokehDOF(CommandBuffer cmd, ref RenderingData renderingData)
            {
                var cameraData = renderingData.cameraData;
                var source = cameraData.renderer.cameraColorTargetHandle;
                var depth = cameraData.renderer.cameraDepthTargetHandle;
                var descriptor = cameraData.cameraTargetDescriptor;

                int width = descriptor.width;
                int height = descriptor.height;
                int downWidth = Mathf.Max(1, Mathf.CeilToInt(width * m_Settings.downsampleScale));
                int downHeight = Mathf.Max(1, Mathf.CeilToInt(height * m_Settings.downsampleScale));

                // Calculate DOF parameters
                float focusDist = m_Volume.focusDistance.value;
                float focusRange = m_Volume.focusRange.value;
                float aperture = m_Volume.aperture.value;
                float focalLength = m_Volume.focalLength.value / 1000f;
                float maxBlur = m_Volume.maxBlurRadius.value;
                float nearScale = m_Volume.nearBlurScale.value;
                float farScale = m_Volume.farBlurScale.value;
                float bokehIntensity = m_Volume.bokehIntensity.value;
                float highlightThreshold = m_Volume.highlightThreshold.value;

                // Calculate CoC scale factor
                float cocScale = focalLength * focalLength / (aperture * Mathf.Max(focusDist - focalLength, 0.001f));

                Vector4 sourceSize = new Vector4(width, height, 1f / width, 1f / height);
                Vector4 downsampledSize = new Vector4(downWidth, downHeight, 1f / downWidth, 1f / downHeight);
                Vector4 dofParams = new Vector4(focusDist, focusRange, cocScale, maxBlur);
                Vector4 blurParams = new Vector4(nearScale, farScale, maxBlur * m_Settings.downsampleScale, 0);
                Vector4 bokehParams = new Vector4(bokehIntensity, highlightThreshold, 0, 0);

                // Set global parameters
                cmd.SetComputeVectorParam(m_ComputeShader, _SourceSize, sourceSize);
                cmd.SetComputeVectorParam(m_ComputeShader, _DownsampledSize, downsampledSize);
                cmd.SetComputeVectorParam(m_ComputeShader, _DOFParams, dofParams);
                cmd.SetComputeVectorParam(m_ComputeShader, _BlurParams, blurParams);
                cmd.SetComputeVectorParam(m_ComputeShader, _BokehParams, bokehParams);

                int threadGroupsX, threadGroupsY;

                //------------------------------------------------------------------
                // Pass 1: Calculate CoC
                //------------------------------------------------------------------
                cmd.SetComputeTextureParam(m_ComputeShader, m_KernelCoC, _DepthTex, depth);
                cmd.SetComputeTextureParam(m_ComputeShader, m_KernelCoC, _CoCTex, m_CoCTexture);

                threadGroupsX = Mathf.CeilToInt(width / 8f);
                threadGroupsY = Mathf.CeilToInt(height / 8f);
                cmd.DispatchCompute(m_ComputeShader, m_KernelCoC, threadGroupsX, threadGroupsY, 1);

                //------------------------------------------------------------------
                // Pass 2: Downsample color and CoC
                //------------------------------------------------------------------
                cmd.SetComputeTextureParam(m_ComputeShader, m_KernelDownsample, _SourceTex, source);
                cmd.SetComputeTextureParam(m_ComputeShader, m_KernelDownsample, _CoCTexRead, m_CoCTexture);
                cmd.SetComputeTextureParam(m_ComputeShader, m_KernelDownsample, _DownsampledColorTex, m_DownsampledColor);
                cmd.SetComputeTextureParam(m_ComputeShader, m_KernelDownsample, _DownsampledCoCTex, m_DownsampledCoC);

                threadGroupsX = Mathf.CeilToInt(downWidth / 8f);
                threadGroupsY = Mathf.CeilToInt(downHeight / 8f);
                cmd.DispatchCompute(m_ComputeShader, m_KernelDownsample, threadGroupsX, threadGroupsY, 1);

                //------------------------------------------------------------------
                // Pass 3: Bokeh Blur (5-tap or 13-tap)
                //------------------------------------------------------------------
                int blurKernel = m_Volume.samplingMode.value == BokehDOFVolume.SamplingMode.FiveTapGolden
                    ? m_KernelBokehBlur5Tap
                    : m_KernelBokehBlur13Tap;

                cmd.SetComputeTextureParam(m_ComputeShader, blurKernel, _DownsampledColorTexRead, m_DownsampledColor);
                cmd.SetComputeTextureParam(m_ComputeShader, blurKernel, _DownsampledCoCTexRead, m_DownsampledCoC);
                cmd.SetComputeTextureParam(m_ComputeShader, blurKernel, _BokehTex, m_BokehTexture);

                cmd.DispatchCompute(m_ComputeShader, blurKernel, threadGroupsX, threadGroupsY, 1);

                //------------------------------------------------------------------
                // Pass 4: Composite
                //------------------------------------------------------------------
                cmd.SetComputeTextureParam(m_ComputeShader, m_KernelComposite, _SourceTex, source);
                cmd.SetComputeTextureParam(m_ComputeShader, m_KernelComposite, _CoCTexRead, m_CoCTexture);
                cmd.SetComputeTextureParam(m_ComputeShader, m_KernelComposite, _BokehTexRead, m_BokehTexture);
                cmd.SetComputeTextureParam(m_ComputeShader, m_KernelComposite, _ResultTex, m_ResultTexture);

                threadGroupsX = Mathf.CeilToInt(width / 8f);
                threadGroupsY = Mathf.CeilToInt(height / 8f);
                cmd.DispatchCompute(m_ComputeShader, m_KernelComposite, threadGroupsX, threadGroupsY, 1);

                //------------------------------------------------------------------
                // Copy result back to camera target
                //------------------------------------------------------------------
                Blitter.BlitCameraTexture(cmd, m_ResultTexture, source);
            }

            public override void OnCameraCleanup(CommandBuffer cmd)
            {
                // Cleanup if needed
            }

            public void Dispose()
            {
                m_CoCTexture?.Release();
                m_DownsampledColor?.Release();
                m_DownsampledCoC?.Release();
                m_BokehTexture?.Release();
                m_ResultTexture?.Release();
            }
        }
    }
}