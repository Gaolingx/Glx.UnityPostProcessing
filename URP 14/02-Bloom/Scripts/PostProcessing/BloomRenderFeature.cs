// BloomRenderFeature.cs
using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace Glx.PostProcess.URP.Runtime
{
    public class BloomRenderFeature : ScriptableRendererFeature
    {
        [Serializable]
        public class BloomSettings
        {
            [Header("Render Settings")]
            public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

            [Header("Compute Shaders")]
            public ComputeShader bloomPrefilterCS;
            public ComputeShader bloomBlurCS;
            public ComputeShader bloomUpsampleCS;

            [Header("Composite Shader")]
            public Shader bloomCompositeShader;
        }

        public BloomSettings settings = new BloomSettings();
        private BloomRenderPass m_RenderPass;

        public override void Create()
        {
            m_RenderPass = new BloomRenderPass(settings);
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (!ValidateSettings())
                return;

            var volume = VolumeManager.instance.stack.GetComponent<BloomVolumeComponent>();
            if (volume == null || !volume.IsActive())
                return;

            if (renderingData.cameraData.cameraType != CameraType.Game &&
                renderingData.cameraData.cameraType != CameraType.SceneView)
                return;

            m_RenderPass.renderPassEvent = settings.renderPassEvent;
            m_RenderPass.ConfigureInput(ScriptableRenderPassInput.Color);
            renderer.EnqueuePass(m_RenderPass);
        }

        private bool ValidateSettings()
        {
            if (settings.bloomPrefilterCS == null)
            {
                Debug.LogWarning("Bloom RenderFeature: Missing Prefilter Compute Shader");
                return false;
            }
            if (settings.bloomBlurCS == null)
            {
                Debug.LogWarning("Bloom RenderFeature: Missing Blur Compute Shader");
                return false;
            }
            if (settings.bloomUpsampleCS == null)
            {
                Debug.LogWarning("Bloom RenderFeature: Missing Upsample Compute Shader");
                return false;
            }
            if (settings.bloomCompositeShader == null)
            {
                Debug.LogWarning("Bloom RenderFeature: Missing Composite Shader");
                return false;
            }
            return true;
        }

        protected override void Dispose(bool disposing)
        {
            m_RenderPass?.Dispose();
        }
    }
}