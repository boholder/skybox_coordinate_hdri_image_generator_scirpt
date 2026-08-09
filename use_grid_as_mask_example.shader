Shader "Custom/use_grid_as_mask_example"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (1, 1, 1, 1.0)
        _MaskTex ("Mask", Cube) = "" {}
    }

    SubShader
    {
        Tags
        {
            "RenderType"="Background" "Queue"="Background" "PreviewType"="SkyBox"
        }

        Pass
        {
            ZWrite Off
            Cull Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            fixed4 _BaseColor;
            samplerCUBE _MaskTex;

            struct appdata
            {
                float4 vertex : POSITION;
                float3 texcoord : TEXCOORD0;
            };

            struct v2f
            {
                float3 texcoord : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.texcoord = v.texcoord;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                fixed4 c = _BaseColor;
                half4 mc = texCUBE(_MaskTex, normalize(i.texcoord));
                c.rgb = lerp(c.rgb, mc.rgb, mc.a);
                return c;
            }
            ENDCG
        }
    }
}