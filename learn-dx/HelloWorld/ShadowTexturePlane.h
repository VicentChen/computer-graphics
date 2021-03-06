#pragma once
#include "Application.h"

class CShadowTexturePlane : public CApplication
{
	Microsoft::WRL::ComPtr<ID3D12Resource> m_VertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW m_VertexBufferView;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_IndexBuffer;
	D3D12_INDEX_BUFFER_VIEW m_IndexBufferView;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_DepthStencilBuffer;

	Microsoft::WRL::ComPtr<ID3D12RootSignature> m_RootSignature;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RenderPipelineState;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ShadowPipelineState;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_Texture;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_ShadowMap;

	D3D12_VIEWPORT m_Viewport;
	D3D12_RECT m_ScissorRect;

	DirectX::XMMATRIX m_ModelMatrix;
	DirectX::XMMATRIX m_ViewMatrix;
	DirectX::XMMATRIX m_ProjectionMatrix;
	float m_FoV = 45;

public:
	void update() override;
	void render() override;

protected:
	void _initPipeline() override;
	void _loadModels() override;
	void _describeAssets() override;
};