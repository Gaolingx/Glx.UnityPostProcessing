// BloomRenderPass.cs
using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace Glx.PostProcess.URP.Runtime
{
    public class BloomRenderPass : ScriptableRenderPass, IDisposable
    {
        private const int k_MaxBloomMipCount = 16;
        private const int k_RTGuardBandSize = 4;
        private const string k_ProfilerTag = "Bloom";

        private readonly BloomRenderFeature.BloomSettings m_Settings;
        private BloomVolumeComponent m_BloomVolume;
        private Material m_CompositeMaterial;

        private RTHandle[] m_MipsDown;
        private RTHandle[] m_MipsUp;
        private RTHandle m_TempTarget;
        private Vector4[] m_BloomMipInfo;
        private int m_BloomMipCount;

        // Compute shader kernels
        private int m_PrefilterKernel;
        private int m_BlurKernel;
        private int m_DownsampleKernel;
        private int m_UpsampleKernel;

        // Shader property IDs - 与Compute Shader中的变量名匹配
        private static class ShaderIDs
        {
            public static readonly int _InputTexture = Shader.PropertyToID("_InputTexture");
            public static readonly int _OutputTexture = Shader.PropertyToID("_OutputTexture");
            public static readonly int _InputLowTexture = Shader.PropertyToID("_InputLowTexture");
            public static readonly int _InputHighTexture = Shader.PropertyToID("_InputHighTexture");
            public static readonly int _TexelSize = Shader.PropertyToID("_TexelSize");
            public static readonly int _BloomThreshold = Shader.PropertyToID("_BloomThreshold");
            public static readonly int _Params = Shader.PropertyToID("_Params");
            public static readonly int _BloomBicubicParams = Shader.PropertyToID("_BloomBicubicParams");

            // Composite shader properties
            public static readonly int _BloomTexture = Shader.PropertyToID("_BloomTexture");
            public static readonly int _BloomParams = Shader.PropertyToID("_BloomParams");
            public static readonly int _BloomTint = Shader.PropertyToID("_BloomTint");
            public static readonly int _ClampMax = Shader.PropertyToID("_ClampMax");
            public static readonly int _BloomDirtTexture = Shader.PropertyToID("_BloomDirtTexture");
            public static readonly int _BloomDirtTileOffset = Shader.PropertyToID("_BloomDirtTileOffset");
            public static readonly int _SourceTex = Shader.PropertyToID("_SourceTex");
        }

        public BloomRenderPass(BloomRenderFeature.BloomSettings settings)
        {
            m_Settings = settings;
            m_MipsDown = new RTHandle[k_MaxBloomMipCount + 1];
            m_MipsUp = new RTHandle[k_MaxBloomMipCount + 1];
            m_BloomMipInfo = new Vector4[k_MaxBloomMipCount + 1];
            profilingSampler = new ProfilingSampler(k_ProfilerTag);
        }

        public void Dispose()
        {
            ReleaseTextures();
            CoreUtils.Destroy(m_CompositeMaterial);
        }

        private void ReleaseTextures()
        {
            for (int i = 0; i <= k_MaxBloomMipCount; i++)
            {
                m_MipsDown[i]?.Release();
                m_MipsUp[i]?.Release();
                m_MipsDown[i] = null;
                m_MipsUp[i] = null;
            }
            m_TempTarget?.Release();
            m_TempTarget = null;
        }

        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            var stack = VolumeManager.instance.stack;
            m_BloomVolume = stack.GetComponent<BloomVolumeComponent>();

            if (m_BloomVolume == null || !m_BloomVolume.IsActive())
                return;

            // Initialize material
            if (m_CompositeMaterial == null && m_Settings.bloomCompositeShader != null)
            {
                m_CompositeMaterial = CoreUtils.CreateEngineMaterial(m_Settings.bloomCompositeShader);
            }

            // Find kernels
            m_PrefilterKernel = m_Settings.bloomPrefilterCS.FindKernel("KMain");
            m_BlurKernel = m_Settings.bloomBlurCS.FindKernel("KMain");
            m_DownsampleKernel = m_Settings.bloomBlurCS.FindKernel("KDownsample");
            m_UpsampleKernel = m_Settings.bloomUpsampleCS.FindKernel("KMain");
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            if (m_BloomVolume == null || !m_BloomVolume.IsActive() || m_CompositeMaterial == null)
                return;

            var cmd = CommandBufferPool.Get(k_ProfilerTag);

            using (new ProfilingScope(cmd, profilingSampler))
            {
                var cameraData = renderingData.cameraData;
                var descriptor = cameraData.cameraTargetDescriptor;
                int width = descriptor.width;
                int height = descriptor.height;

                var source = cameraData.renderer.cameraColorTargetHandle;

                // Setup and execute bloom
                SetupBloomMips(width, height);
                ExecuteBloom(cmd, source, width, height);

                // Composite bloom to screen
                CompositeBloom(cmd, source, width, height);
            }

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        private void SetupBloomMips(int width, int height)
        {
            ReleaseTextures();

            // Calculate scale based on resolution setting
            float scaleW = 1f / ((int)m_BloomVolume.resolution.value / 2f);
            float scaleH = 1f / ((int)m_BloomVolume.resolution.value / 2f);

            // Use full res for very small screens
            if (width < 800 || height < 450)
            {
                scaleW = 1.0f;
                scaleH = 1.0f;
            }

            // Calculate iteration count
            int maxSize = Mathf.Max(width, height);
            int iterations = Mathf.FloorToInt(Mathf.Log(maxSize, 2f) - 2 - (m_BloomVolume.resolution.value == BloomResolution.Half ? 0 : 1));
            m_BloomMipCount = Mathf.Clamp(iterations, 1, k_MaxBloomMipCount);

            // Allocate mip textures
            for (int i = 0; i < m_BloomMipCount; i++)
            {
                float p = 1f / Mathf.Pow(2f, i + 1f);
                float sw = scaleW * p;
                float sh = scaleH * p;
                int pw = Mathf.Max(1, Mathf.RoundToInt(sw * width));
                int ph = Mathf.Max(1, Mathf.RoundToInt(sh * height));

                m_BloomMipInfo[i] = new Vector4(pw, ph, sw, sh);

                var desc = new RenderTextureDescriptor(pw, ph, RenderTextureFormat.DefaultHDR, 0)
                {
                    enableRandomWrite = true,
                    useMipMap = false,
                    sRGB = false
                };

                m_MipsDown[i] = RTHandles.Alloc(desc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: $"_BloomMipDown{i}");

                if (i != 0)
                {
                    m_MipsUp[i] = RTHandles.Alloc(desc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: $"_BloomMipUp{i}");
                }
            }

            // Mip 0 up texture (used as final output)
            {
                int pw = (int)m_BloomMipInfo[0].x;
                int ph = (int)m_BloomMipInfo[0].y;
                var desc = new RenderTextureDescriptor(pw, ph, RenderTextureFormat.DefaultHDR, 0)
                {
                    enableRandomWrite = true,
                    useMipMap = false,
                    sRGB = false
                };
                m_MipsUp[0] = RTHandles.Alloc(desc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_BloomMipUp0");
            }

            // Temp target for final composite
            {
                var desc = new RenderTextureDescriptor(width, height, RenderTextureFormat.DefaultHDR, 0)
                {
                    enableRandomWrite = false,
                    useMipMap = false,
                    sRGB = false
                };
                m_TempTarget = RTHandles.Alloc(desc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_BloomTempTarget");
            }
        }

        private void ExecuteBloom(CommandBuffer cmd, RTHandle source, int width, int height)
        {
            // Calculate threshold parameters
            const float k_Softness = 0.5f;
            float threshold = Mathf.GammaToLinearSpace(m_BloomVolume.threshold.value);
            float knee = threshold * k_Softness + 1e-5f;
            Vector4 thresholdParams = new Vector4(threshold, threshold - knee, knee * 2f, 0.25f / knee);

            // Setup prefilter keywords
            SetKeyword(m_Settings.bloomPrefilterCS, "HIGH_QUALITY", m_BloomVolume.highQualityPrefiltering.value);
            SetKeyword(m_Settings.bloomPrefilterCS, "LOW_QUALITY", !m_BloomVolume.highQualityPrefiltering.value);

            // === Prefilter Pass ===
            {
                int mipWidth = (int)m_BloomMipInfo[0].x;
                int mipHeight = (int)m_BloomMipInfo[0].y;

                cmd.SetComputeTextureParam(m_Settings.bloomPrefilterCS, m_PrefilterKernel, ShaderIDs._InputTexture, source);
                cmd.SetComputeTextureParam(m_Settings.bloomPrefilterCS, m_PrefilterKernel, ShaderIDs._OutputTexture, m_MipsUp[0]);
                cmd.SetComputeVectorParam(m_Settings.bloomPrefilterCS, ShaderIDs._TexelSize, new Vector4(mipWidth, mipHeight, 1f / mipWidth, 1f / mipHeight));
                cmd.SetComputeVectorParam(m_Settings.bloomPrefilterCS, ShaderIDs._BloomThreshold, thresholdParams);
                cmd.SetComputeFloatParam(m_Settings.bloomPrefilterCS, ShaderIDs._ClampMax, m_BloomVolume.clamp.value);

                DispatchCompute(cmd, m_Settings.bloomPrefilterCS, m_PrefilterKernel, mipWidth, mipHeight, width, height);
            }

            // === Initial Blur Pass ===
            {
                int mipWidth = (int)m_BloomMipInfo[0].x;
                int mipHeight = (int)m_BloomMipInfo[0].y;

                cmd.SetComputeTextureParam(m_Settings.bloomBlurCS, m_BlurKernel, ShaderIDs._InputTexture, m_MipsUp[0]);
                cmd.SetComputeTextureParam(m_Settings.bloomBlurCS, m_BlurKernel, ShaderIDs._OutputTexture, m_MipsDown[0]);
                cmd.SetComputeVectorParam(m_Settings.bloomBlurCS, ShaderIDs._TexelSize, new Vector4(mipWidth, mipHeight, 1f / mipWidth, 1f / mipHeight));

                DispatchCompute(cmd, m_Settings.bloomBlurCS, m_BlurKernel, mipWidth, mipHeight, width, height);
            }

            // === Downsample Pyramid ===
            for (int i = 0; i < m_BloomMipCount - 1; i++)
            {
                int mipWidth = (int)m_BloomMipInfo[i + 1].x;
                int mipHeight = (int)m_BloomMipInfo[i + 1].y;

                cmd.SetComputeTextureParam(m_Settings.bloomBlurCS, m_DownsampleKernel, ShaderIDs._InputTexture, m_MipsDown[i]);
                cmd.SetComputeTextureParam(m_Settings.bloomBlurCS, m_DownsampleKernel, ShaderIDs._OutputTexture, m_MipsDown[i + 1]);
                cmd.SetComputeVectorParam(m_Settings.bloomBlurCS, ShaderIDs._TexelSize, new Vector4(mipWidth, mipHeight, 1f / mipWidth, 1f / mipHeight));

                DispatchCompute(cmd, m_Settings.bloomBlurCS, m_DownsampleKernel, mipWidth, mipHeight, width, height);
            }

            // === Upsample & Combine ===
            bool highQualityFiltering = m_BloomVolume.highQualityFiltering.value;
            // Reduce quality for small screens to avoid artifacts
            if (width < 800 || height < 450) highQualityFiltering = false;

            SetKeyword(m_Settings.bloomUpsampleCS, "HIGH_QUALITY", highQualityFiltering);
            SetKeyword(m_Settings.bloomUpsampleCS, "LOW_QUALITY", !highQualityFiltering);

            float scatter = Mathf.Lerp(0.05f, 0.95f, m_BloomVolume.scatter.value);

            for (int i = m_BloomMipCount - 2; i >= 0; i--)
            {
                var low = (i == m_BloomMipCount - 2) ? m_MipsDown : m_MipsUp;
                var srcLow = low[i + 1];
                var srcHigh = m_MipsDown[i];
                var dst = m_MipsUp[i];

                int highWidth = (int)m_BloomMipInfo[i].x;
                int highHeight = (int)m_BloomMipInfo[i].y;
                int lowWidth = (int)m_BloomMipInfo[i + 1].x;
                int lowHeight = (int)m_BloomMipInfo[i + 1].y;

                cmd.SetComputeTextureParam(m_Settings.bloomUpsampleCS, m_UpsampleKernel, ShaderIDs._InputLowTexture, srcLow);
                cmd.SetComputeTextureParam(m_Settings.bloomUpsampleCS, m_UpsampleKernel, ShaderIDs._InputHighTexture, srcHigh);
                cmd.SetComputeTextureParam(m_Settings.bloomUpsampleCS, m_UpsampleKernel, ShaderIDs._OutputTexture, dst);
                cmd.SetComputeVectorParam(m_Settings.bloomUpsampleCS, ShaderIDs._Params, new Vector4(scatter, 0f, 0f, 0f));
                cmd.SetComputeVectorParam(m_Settings.bloomUpsampleCS, ShaderIDs._BloomBicubicParams, new Vector4(lowWidth, lowHeight, 1f / lowWidth, 1f / lowHeight));
                cmd.SetComputeVectorParam(m_Settings.bloomUpsampleCS, ShaderIDs._TexelSize, new Vector4(highWidth, highHeight, 1f / highWidth, 1f / highHeight));

                DispatchCompute(cmd, m_Settings.bloomUpsampleCS, m_UpsampleKernel, highWidth, highHeight, width, height);
            }
        }

        private void CompositeBloom(CommandBuffer cmd, RTHandle source, int width, int height)
        {
            // Calculate bloom parameters
            float intensity = Mathf.Pow(2f, m_BloomVolume.intensity.value) - 1f;
            var tint = m_BloomVolume.tint.value.linear;
            var luma = ColorUtils.Luminance(tint);
            tint = luma > 0f ? tint * (1f / luma) : Color.white;

            // Dirt texture setup
            var dirtTexture = m_BloomVolume.dirtTexture.value == null ? Texture2D.blackTexture : m_BloomVolume.dirtTexture.value;
            int dirtEnabled = m_BloomVolume.dirtTexture.value != null && m_BloomVolume.dirtIntensity.value > 0f ? 1 : 0;
            float dirtIntensity = m_BloomVolume.dirtIntensity.value * intensity;

            // Calculate dirt tile offset
            float dirtRatio = (float)dirtTexture.width / (float)dirtTexture.height;
            float screenRatio = (float)width / (float)height;
            var dirtTileOffset = new Vector4(1f, 1f, 0f, 0f);

            if (dirtRatio > screenRatio)
            {
                dirtTileOffset.x = screenRatio / dirtRatio;
                dirtTileOffset.z = (1f - dirtTileOffset.x) * 0.5f;
            }
            else if (screenRatio > dirtRatio)
            {
                dirtTileOffset.y = dirtRatio / screenRatio;
                dirtTileOffset.w = (1f - dirtTileOffset.y) * 0.5f;
            }

            // Set material properties
            m_CompositeMaterial.SetTexture(ShaderIDs._SourceTex, source);
            m_CompositeMaterial.SetTexture(ShaderIDs._BloomTexture, m_MipsUp[0]);
            m_CompositeMaterial.SetVector(ShaderIDs._BloomParams, new Vector4(intensity, dirtIntensity, 1f, dirtEnabled));
            m_CompositeMaterial.SetVector(ShaderIDs._BloomTint, (Vector4)tint);
            m_CompositeMaterial.SetTexture(ShaderIDs._BloomDirtTexture, dirtTexture);
            m_CompositeMaterial.SetVector(ShaderIDs._BloomDirtTileOffset, dirtTileOffset);

            // Blit with composite
            Blitter.BlitCameraTexture(cmd, source, m_TempTarget, m_CompositeMaterial, 0);
            Blitter.BlitCameraTexture(cmd, m_TempTarget, source);
        }

        private void DispatchCompute(CommandBuffer cmd, ComputeShader cs, int kernel, int targetWidth, int targetHeight, int sourceWidth, int sourceHeight)
        {
            int w = targetWidth;
            int h = targetHeight;

            // Guard bands to prevent edge artifacts
            if (w < sourceWidth && w % 8 < k_RTGuardBandSize)
                w += k_RTGuardBandSize;
            if (h < sourceHeight && h % 8 < k_RTGuardBandSize)
                h += k_RTGuardBandSize;

            cmd.DispatchCompute(cs, kernel, (w + 7) / 8, (h + 7) / 8, 1);
        }

        private void SetKeyword(ComputeShader cs, string keyword, bool enabled)
        {
            if (enabled)
                cs.EnableKeyword(keyword);
            else
                cs.DisableKeyword(keyword);
        }

        public override void OnCameraCleanup(CommandBuffer cmd)
        {
            // Textures released in next frame setup or dispose
        }
    }
}