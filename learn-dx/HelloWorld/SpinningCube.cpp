#include "SpinningCube.h"
#include "Igniter.h"
#include <chrono>

using namespace Microsoft::WRL;
using namespace DirectX;
using namespace config;

// Vertex data for a colored cube.
struct VertexPosColor
{
	XMFLOAT3 Position;
	XMFLOAT3 Color;
};

static VertexPosColor Vertices[8] = {
	{ XMFLOAT3(-1.0f, -1.0f, -1.0f), XMFLOAT3(0.0f, 0.0f, 0.0f) }, // 0
	{ XMFLOAT3(-1.0f,  1.0f, -1.0f), XMFLOAT3(0.0f, 1.0f, 0.0f) }, // 1
	{ XMFLOAT3( 1.0f,  1.0f, -1.0f), XMFLOAT3(1.0f, 1.0f, 0.0f) }, // 2
	{ XMFLOAT3( 1.0f, -1.0f, -1.0f), XMFLOAT3(1.0f, 0.0f, 0.0f) }, // 3
	{ XMFLOAT3(-1.0f, -1.0f,  1.0f), XMFLOAT3(0.0f, 0.0f, 1.0f) }, // 4
	{ XMFLOAT3(-1.0f,  1.0f,  1.0f), XMFLOAT3(0.0f, 1.0f, 1.0f) }, // 5
	{ XMFLOAT3( 1.0f,  1.0f,  1.0f), XMFLOAT3(1.0f, 1.0f, 1.0f) }, // 6
	{ XMFLOAT3( 1.0f, -1.0f,  1.0f), XMFLOAT3(1.0f, 0.0f, 1.0f) }  // 7
};

static WORD Indicies[36] =
{
	0, 1, 2, 0, 2, 3,
	4, 6, 5, 4, 7, 6,
	4, 5, 1, 4, 1, 0,
	3, 2, 6, 3, 6, 7,
	1, 5, 6, 1, 6, 2,
	4, 0, 3, 4, 3, 7
};

void CSpinningCube::start()
{
	createDescriptorHeap(	0,0,0,0,1);
	
	auto Device = CIgniter::get()->fetchDevice();
	auto CommandQueue = CIgniter::get()->fetchCommandQueue(D3D12_COMMAND_LIST_TYPE_COPY);
	auto CommandList = CommandQueue->createCommandList();
	
	// Upload vertex buffer
	UINT64 VertexBufferSize = _countof(Vertices) * sizeof(VertexPosColor);
	ComPtr<ID3D12Resource> VertexIntermediateBuffer;
	D3D12_SUBRESOURCE_DATA VertexSubresourceData = { Vertices, VertexBufferSize, VertexBufferSize };
	debug::check(Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(VertexBufferSize), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_VertexBuffer)));
	debug::check(Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(VertexBufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&VertexIntermediateBuffer)));
	UpdateSubresources(CommandList.Get(), m_VertexBuffer.Get(), VertexIntermediateBuffer.Get(), 0, 0, 1, &VertexSubresourceData);
	m_VertexBufferView.BufferLocation = m_VertexBuffer->GetGPUVirtualAddress();
	m_VertexBufferView.SizeInBytes = sizeof(Vertices);
	m_VertexBufferView.StrideInBytes = sizeof(VertexPosColor);
	
	// Upload index buffer
	UINT64 IndexBufferSize = _countof(Indicies) * sizeof(WORD);
	ComPtr<ID3D12Resource> IndexIntermediateBuffer;
	D3D12_SUBRESOURCE_DATA IndexSubresourceData = { Indicies, IndexBufferSize, IndexBufferSize };
	debug::check(Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(IndexBufferSize), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_IndexBuffer)));
	debug::check(Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(IndexBufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&IndexIntermediateBuffer)));
	UpdateSubresources(CommandList.Get(), m_IndexBuffer.Get(), IndexIntermediateBuffer.Get(), 0, 0, 1, &IndexSubresourceData);
	m_IndexBufferView.BufferLocation = m_IndexBuffer->GetGPUVirtualAddress();
	m_IndexBufferView.Format = DXGI_FORMAT_R16_UINT;
	m_IndexBufferView.SizeInBytes = sizeof(Indicies);

	// Create depth and stencil buffer	
	D3D12_CLEAR_VALUE OptimizedDepthClearValue = {};
	OptimizedDepthClearValue.Format = DXGI_FORMAT_D32_FLOAT;
	OptimizedDepthClearValue.DepthStencil = { 1.0f, 0 };
	debug::check(Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, window::Width, window::Height, 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), D3D12_RESOURCE_STATE_DEPTH_WRITE, &OptimizedDepthClearValue, IID_PPV_ARGS(&m_DepthStencilBuffer)));
	D3D12_DEPTH_STENCIL_VIEW_DESC DepthStencilViewDesc = {};
	DepthStencilViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
	DepthStencilViewDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	DepthStencilViewDesc.Texture2D.MipSlice = 0;
	DepthStencilViewDesc.Flags = D3D12_DSV_FLAG_NONE;
	Device->CreateDepthStencilView(m_DepthStencilBuffer.Get(), &DepthStencilViewDesc, m_DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

#if defined(_DEBUG)
	// Enable better shader debugging with the graphics debugging tools.
	UINT CompileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
	UINT CompileFlags = 0;
#endif
	
	// Load vertex shader
	ComPtr<ID3DBlob> VertexShaderBlob;
	debug::check(D3DCompileFromFile(L"SpinningCubeVS.hlsl", nullptr, nullptr, shader::EntryPoint.c_str(), shader::VSTarget.c_str(), CompileFlags, 0, &VertexShaderBlob, nullptr));
	// Loda pixel shader
	ComPtr<ID3DBlob> PixelShaderBlob;
	debug::check(D3DCompileFromFile(L"SpinningCubePS.hlsl", nullptr, nullptr, shader::EntryPoint.c_str(), shader::PSTarget.c_str(), CompileFlags, 0, &PixelShaderBlob , nullptr));

	// Vertex input layout
	D3D12_INPUT_ELEMENT_DESC VertexInputDesc[] = {
		{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		{"COLOR", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
	};
	
	D3D12_FEATURE_DATA_ROOT_SIGNATURE RootSignatureFeature = { D3D_ROOT_SIGNATURE_VERSION_1_1 };
	if (FAILED(Device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &RootSignatureFeature, sizeof(RootSignatureFeature))))
	{
		logger::warning("Feature D3D_ROOT_SIGNATURE_VERSION_1_1 not support");
		RootSignatureFeature.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
	}
	D3D12_ROOT_SIGNATURE_FLAGS RootSignatureFlags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
		D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
		D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
		D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
		D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS;

	CD3DX12_ROOT_PARAMETER1 RootParameter;
	RootParameter.InitAsConstants(sizeof(XMMATRIX) / 4, 0, 0, D3D12_SHADER_VISIBILITY_VERTEX);
	CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC RootSignatureDesc;
	RootSignatureDesc.Init_1_1(1, &RootParameter, 0, nullptr, RootSignatureFlags);

	ComPtr<ID3DBlob> RootSignatureBlob;
	ComPtr<ID3DBlob> ErrorBlob;
	debug::check(D3DX12SerializeVersionedRootSignature(&RootSignatureDesc, RootSignatureFeature.HighestVersion, &RootSignatureBlob, &ErrorBlob));
	debug::check(Device->CreateRootSignature(0, RootSignatureBlob->GetBufferPointer(), RootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&m_RootSignature)));

	//struct SPipelineStateStream
	//{
	//	CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
	//	CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
	//	CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
	//	CD3DX12_PIPELINE_STATE_STREAM_VS VS;
	//	CD3DX12_PIPELINE_STATE_STREAM_PS PS;
	//	CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
	//	CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
	//} PipelineStateStream;

	//D3D12_RT_FORMAT_ARRAY RTVFormats = {};
	//RTVFormats.NumRenderTargets = 1;
	//RTVFormats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;

	//PipelineStateStream.pRootSignature = m_RootSignature.Get();
	//PipelineStateStream.InputLayout = { VertexInputDesc, _countof(VertexInputDesc) };
	//PipelineStateStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	//PipelineStateStream.VS = CD3DX12_SHADER_BYTECODE(VertexShaderBlob.Get());
	//PipelineStateStream.PS = CD3DX12_SHADER_BYTECODE(PixelShaderBlob.Get());
	//PipelineStateStream.DSVFormat = DXGI_FORMAT_D32_FLOAT;
	//PipelineStateStream.RTVFormats = RTVFormats;

	//D3D12_PIPELINE_STATE_STREAM_DESC PipelineStateStreamDesc = { sizeof(PipelineStateStream), &PipelineStateStream };
	//debug::check(Device->CreatePipelineState(&PipelineStateStreamDesc, IID_PPV_ARGS(&m_PipelineState)));

	D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
	psoDesc.pRootSignature = m_RootSignature.Get();
	psoDesc.InputLayout = { VertexInputDesc, _countof(VertexInputDesc) };
	psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	psoDesc.VS = CD3DX12_SHADER_BYTECODE(VertexShaderBlob.Get());
	psoDesc.PS = CD3DX12_SHADER_BYTECODE(PixelShaderBlob.Get());
	psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
	psoDesc.NumRenderTargets = 1;
	psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	//psoDesc.DepthStencilState.DepthEnable = TRUE;
	//psoDesc.DepthStencilState.StencilEnable = TRUE;
	psoDesc.SampleMask = UINT_MAX;
	psoDesc.SampleDesc.Count = 1;
	debug::check(Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_PipelineState)));
	
	auto FenceValue = CommandQueue->executeCommandList(CommandList);
	CommandQueue->wait4Fence(FenceValue);

	m_Viewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(window::Width), static_cast<float>(window::Height));
	m_ScissorRect = CD3DX12_RECT(0, 0, LONG_MAX, LONG_MAX);

	CommandQueue->Flush();
}

void CSpinningCube::update()
{
	static UINT64 FrameCount = 0;
	FrameCount++;
	
	double Angle = FrameCount / 100.0f; // 9000 frames - 90 degree
	const XMVECTOR RotationAxis = XMVectorSet(0, 1, 1, 0);
	m_ModelMatrix = XMMatrixRotationAxis(RotationAxis, Angle);
	
	const XMVECTOR eyePosition = XMVectorSet(0, 0, -10, 1);
	const XMVECTOR focusPoint = XMVectorSet(0, 0, 0, 1);
	const XMVECTOR upDirection = XMVectorSet(0, 1, 0, 0);
	m_ViewMatrix = XMMatrixLookAtLH(eyePosition, focusPoint, upDirection);

	// Update the projection matrix.
	float aspectRatio = window::Width / static_cast<float>(window::Height);
	m_ProjectionMatrix = XMMatrixPerspectiveFovLH(XMConvertToRadians(m_FoV), aspectRatio, 0.1f, 100.0f);
}

void CSpinningCube::render()
{
	auto pIgniter = CIgniter::get();
	auto pCommandQueue = pIgniter->fetchCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto SwapChain = pIgniter->fetchSwapChain();
	auto CommandList = pCommandQueue->createCommandList();

	auto RTV = pIgniter->fetchCurrentRTV();
	auto DSV = m_DSVDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
	auto BackBuffer = pIgniter->fetchCurrentBackBuffer();

	// clear render target
	{
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(BackBuffer.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
		CommandList->ResourceBarrier(1, &barrier);
		CommandList->ClearRenderTargetView(RTV, window::ClearColor, 0, nullptr);
		CommandList->ClearDepthStencilView(DSV, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
	}

	CommandList->SetPipelineState(m_PipelineState.Get());
	CommandList->SetGraphicsRootSignature(m_RootSignature.Get());
	CommandList->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	CommandList->IASetVertexBuffers(0, 1, &m_VertexBufferView);
	CommandList->IASetIndexBuffer(&m_IndexBufferView);
	CommandList->RSSetViewports(1, &m_Viewport);
	CommandList->RSSetScissorRects(1, &m_ScissorRect);
	CommandList->OMSetRenderTargets(1, &RTV, FALSE, &DSV);

	XMMATRIX MVPMatrix = XMMatrixMultiply(m_ModelMatrix, m_ViewMatrix);
	MVPMatrix = XMMatrixMultiply(MVPMatrix, m_ProjectionMatrix);
	CommandList->SetGraphicsRoot32BitConstants(0, sizeof(XMMATRIX) / 4, &MVPMatrix, 0);
	CommandList->DrawIndexedInstanced(_countof(Indicies), 1, 0, 0, 0);
	
	// present
	{
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(BackBuffer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
		CommandList->ResourceBarrier(1, &barrier);
		UINT64 FenceValue = pCommandQueue->executeCommandList(CommandList);
		debug::check(SwapChain->Present(0, 0));
		pCommandQueue->wait4Fence(FenceValue);
	}
}

void CSpinningCube::shutdown()
{
}
