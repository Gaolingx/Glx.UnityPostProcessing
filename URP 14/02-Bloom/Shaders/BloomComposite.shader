// BloomComposite.shader
Shader "Hidden/CustomPostProcess/Bloom/BloomComposite"
{
    HLSLINCLUDE
    
    #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
    
    TEXTURE2D(_SourceTex);
    TEXTURE2D(_BloomTexture);
    TEXTURE2D(_BloomDirtTexture);
    
    SAMPLER(sampler_LinearClamp);
    
    float4 _BloomParams;         // x: intensity, y: dirt intensity, z: unused, w: dirt enabled
    float4 _BloomTint;
    float4 _BloomDirtTileOffset; // xy: tiling, zw: offset
    
    #define BloomIntensity      _BloomParams.x
    #define BloomDirtIntensity  _BloomParams.y
    #define BloomDirtEnabled    _BloomParams.w
    
    struct Attributes
    {
        uint vertexID : SV_VertexID;
    };
    
    struct Varyings
    {
        float4 positionCS : SV_POSITION;
        float2 texcoord   : TEXCOORD0;
    };
    
    Varyings Vert(Attributes input)
    {
        Varyings output;
        output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
        output.texcoord = GetFullScreenTriangleTexCoord(input.vertexID);
        return output;
    }
    
    float4 Frag(Varyings input) : SV_Target
    {
        float2 uv = input.texcoord;
        
        float4 source = SAMPLE_TEXTURE2D(_SourceTex, sampler_LinearClamp, uv);
        float3 bloom = SAMPLE_TEXTURE2D(_BloomTexture, sampler_LinearClamp, uv).rgb;
        
        // Apply bloom tint and intensity
        bloom *= _BloomTint.rgb * BloomIntensity;
        
        // Apply lens dirt
        if (BloomDirtEnabled > 0.5)
        {
            float2 dirtUV = uv * _BloomDirtTileOffset.xy + _BloomDirtTileOffset.zw;
            float3 dirt = SAMPLE_TEXTURE2D(_BloomDirtTexture, sampler_LinearClamp, dirtUV).rgb;
            bloom += dirt * bloom * BloomDirtIntensity;
        }
        
        // Additive blend
        float3 result = source.rgb + bloom;
        
        return float4(result, source.a);
    }
    
    ENDHLSL
    
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }
        LOD 100
        ZTest Always ZWrite Off Cull Off
        
        Pass
        {
            Name "Bloom Composite"
            
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            ENDHLSL
        }
    }
}