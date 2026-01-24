// BloomVolumeComponent.cs
using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace Glx.PostProcess.URP.Runtime
{
    public enum BloomResolution : int
    {
        Quarter = 4,
        Half = 2
    }

    [Serializable, VolumeComponentMenuForRenderPipeline("Glx Post Process/Bloom", typeof(UniversalRenderPipeline))]
    public sealed class BloomVolumeComponent : VolumeComponent, IPostProcessComponent
    {
        [Header("Bloom")]
        [Tooltip("Set the level of brightness to filter out pixels under this level. This value is expressed in gamma-space.")]
        public MinFloatParameter threshold = new MinFloatParameter(0.9f, 0f);

        [Tooltip("Controls the strength of the bloom filter.")]
        public MinFloatParameter intensity = new MinFloatParameter(0f, 0f);

        [Tooltip("Controls the extent of the veiling effect.")]
        public ClampedFloatParameter scatter = new ClampedFloatParameter(0.7f, 0f, 1f);

        [Tooltip("Specifies the tint of the bloom filter.")]
        public ColorParameter tint = new ColorParameter(Color.white, false, false, true);

        [Tooltip("Clamps pixels to control the bloom amount. Lower values reduce the bloom intensity on overly bright areas.")]
        public MinFloatParameter clamp = new MinFloatParameter(65472f, 0f);

        [Header("Lens Dirt")]
        [Tooltip("Specifies a Texture to add smudges or dust to the bloom effect.")]
        public Texture2DParameter dirtTexture = new Texture2DParameter(null);

        [Tooltip("Controls the strength of the lens dirt.")]
        public MinFloatParameter dirtIntensity = new MinFloatParameter(0f, 0f);

        [Header("Advanced")]
        [Tooltip("Specifies the resolution at which the effect is processed.")]
        public BloomResolutionParameter resolution = new BloomResolutionParameter(BloomResolution.Half);

        [Tooltip("When enabled, bloom uses multiple bilinear samples for the prefiltering pass.")]
        public BoolParameter highQualityPrefiltering = new BoolParameter(false);

        [Tooltip("When enabled, bloom uses bicubic sampling instead of bilinear sampling for the upsampling passes.")]
        public BoolParameter highQualityFiltering = new BoolParameter(true);

        public bool IsActive() => intensity.value > 0f;
        public bool IsTileCompatible() => false;
    }

    [Serializable]
    public sealed class BloomResolutionParameter : VolumeParameter<BloomResolution>
    {
        public BloomResolutionParameter(BloomResolution value, bool overrideState = false)
            : base(value, overrideState) { }
    }
}