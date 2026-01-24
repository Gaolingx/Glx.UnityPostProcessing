// BokehDOFVolume.cs
using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace Glx.PostProcess.URP.Runtime
{
    [Serializable, VolumeComponentMenu("Custom Post-processing/Bokeh DOF")]
    public sealed class BokehDOFVolume : VolumeComponent, IPostProcessComponent
    {
        public enum SamplingMode
        {
            FiveTapGolden = 0,
            ThirteenTapDisc = 1
        }

        [Serializable]
        public sealed class SamplingModeParameter : VolumeParameter<SamplingMode>
        {
            public SamplingModeParameter(SamplingMode value, bool overrideState = false)
                : base(value, overrideState) { }
        }

        [Header("基础设置")]
        [Tooltip("采样模式")]
        public SamplingModeParameter samplingMode = new SamplingModeParameter(SamplingMode.ThirteenTapDisc);

        [Header("焦点设置")]
        [Tooltip("焦点距离（米）")]
        public ClampedFloatParameter focusDistance = new ClampedFloatParameter(10f, 0.1f, 100f);

        [Tooltip("焦点范围（过渡区域大小）")]
        public ClampedFloatParameter focusRange = new ClampedFloatParameter(5f, 0.1f, 50f);

        [Header("光学参数")]
        [Tooltip("光圈大小 (f-stop)，值越小模糊越强")]
        public ClampedFloatParameter aperture = new ClampedFloatParameter(5.6f, 1f, 22f);

        [Tooltip("焦距 (mm)")]
        public ClampedFloatParameter focalLength = new ClampedFloatParameter(50f, 10f, 300f);

        [Header("模糊控制")]
        [Tooltip("最大模糊半径（像素）")]
        public ClampedFloatParameter maxBlurRadius = new ClampedFloatParameter(8f, 0f, 32f);

        [Tooltip("近景模糊强度")]
        public ClampedFloatParameter nearBlurScale = new ClampedFloatParameter(1f, 0f, 2f);

        [Tooltip("远景模糊强度")]
        public ClampedFloatParameter farBlurScale = new ClampedFloatParameter(1f, 0f, 2f);

        [Header("高级设置")]
        [Tooltip("Bokeh亮度增强")]
        public ClampedFloatParameter bokehIntensity = new ClampedFloatParameter(1f, 0.5f, 3f);

        [Tooltip("高光阈值")]
        public ClampedFloatParameter highlightThreshold = new ClampedFloatParameter(1f, 0f, 5f);

        public bool IsActive() => maxBlurRadius.value > 0f && active;

        public bool IsTileCompatible() => false;
    }
}