struct PixelShaderInput
{
	float2 TexCoord : TEXCOORD;
	float4 Color    : FRAGCOLOR;
};

Texture2D Texture : register(t0);
SamplerState Sampler : register(s0);

float4 main(PixelShaderInput IN) : SV_Target
{
	//return IN.Color;
	return Texture.Sample(Sampler, IN.TexCoord);
}