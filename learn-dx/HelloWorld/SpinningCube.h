#pragma once
#include "Application.h"

class CSpinningCube : public CApplication
{
	Microsoft::WRL::ComPtr<ID3D12Resource> m_VertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW m_VertexBufferView;
	
	Microsoft::WRL::ComPtr<ID3D12Resource> m_IndexBuffer;
	D3D12_INDEX_BUFFER_VIEW m_IndexBufferView;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_DepthStencilBuffer;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_DepthStencilHeap;

	Microsoft::WRL::ComPtr<ID3D12RootSignature> m_RootSignature;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> m_PipelineState;

	D3D12_VIEWPORT m_Viewport;
	D3D12_RECT m_ScissorRect;
	
	DirectX::XMMATRIX m_ModelMatrix;
	DirectX::XMMATRIX m_ViewMatrix;
	DirectX::XMMATRIX m_ProjectionMatrix;
	float m_FoV = 45;
	
public:
	void start() override;
	void update() override;
	void render() override;
	void shutdown() override;
};