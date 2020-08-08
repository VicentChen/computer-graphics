struct PixelShaderInput
{
	float2 TexCoord : TEXCOORD;
	float4 Color    : FRAGCOLOR;
};

Texture2D Texture : register(t0);
Texture2D ShadowMap : register(t1);
SamplerState Sampler : register(s0);

float4 main(PixelShaderInput IN) : SV_Target
{
	//return IN.Color;
	//return Texture.Sample(Sampler, IN.TexCoord);
	float depth = (ShadowMap.Sample(Sampler, IN.TexCoord) - 0.9) * 10;
	return float4(depth, depth, depth, 1.0f);
}