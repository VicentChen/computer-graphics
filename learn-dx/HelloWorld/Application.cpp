#include "Igniter.h"
#include "Application.h"
#include "dxcapi.use.h"
#include <fstream>
#include <sstream>

template<class BlotType> std::string convertBlobToString(BlotType* pBlob);

static dxc::DxcDllSupport DxcDllHelper;

void CApplication::SDescriptorHeapAttributes::init(D3D12_DESCRIPTOR_HEAP_DESC& vDesc, ID3D12DescriptorHeap* vHeap)
{
	DescriptorSize = CIgniter::get()->getDescriptorHandleIncrementSize(vDesc.Type);
	Desc = vDesc;
	CPUHandle = vHeap->GetCPUDescriptorHandleForHeapStart();
	GPUHandle = vHeap->GetGPUDescriptorHandleForHeapStart();
}

void CApplication::start()
{
	_initPipeline();
	_loadModels();
	_describeAssets();
}

void CApplication::render()
{
	auto pIgniter = CIgniter::get();
	auto pCommandQueue = pIgniter->fetchCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto SwapChain = pIgniter->fetchSwapChain();
	auto CommandList = pCommandQueue->createCommandList();

	auto RTV = pIgniter->fetchCurrentRTV();
	auto BackBuffer = pIgniter->fetchCurrentBackBuffer();

	// clear render target
	{
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(BackBuffer.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
		CommandList->ResourceBarrier(1, &barrier);
		CommandList->ClearRenderTargetView(RTV, config::window::ClearColor, 0, nullptr);
	}
	
	// present
	{
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(BackBuffer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
		CommandList->ResourceBarrier(1, &barrier);
		UINT64 FenceValue = pCommandQueue->executeCommandList(CommandList);
		debug::check(SwapChain->Present(0, 0));
		pCommandQueue->wait4Fence(FenceValue);
	}
}

//void CApplication::uploadBuffer(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2>& vCommandList, Microsoft::WRL::ComPtr<ID3D12Resource>& vResource, D3D12_RESOURCE_DESC& vDesc, D3D12_SUBRESOURCE_DATA& vSubresourceData)
//{
//	auto pDevice = CIgniter::get()->fetchDevice();
//	debug::check(pDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &vDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&vResource)));
//
//	Microsoft::WRL::ComPtr<ID3D12Resource> IntermediateResource;
//	UINT64 BufferSize = GetRequiredIntermediateSize(vResource.Get(), 0, 1);
//	debug::check(pDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(BufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&IntermediateResource)));
//
//	UpdateSubresources(vCommandList.Get(), vResource.Get(), IntermediateResource.Get(), 0, 0, 1, &vSubresourceData);
//}

void CApplication::createDescriptorHeap(int vCBVDescriptorCount, int vSRVDescriptorNum, int vUAVDescriptorNum, int vSamplerNum, int vDSVDescriptorNum) {
	auto pDevice = CIgniter::get()->fetchDevice();
	
	D3D12_DESCRIPTOR_HEAP_DESC CBVSRVUAVHeapDesc = {};
	CBVSRVUAVHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	CBVSRVUAVHeapDesc.NumDescriptors = vCBVDescriptorCount + vSRVDescriptorNum + vUAVDescriptorNum;
	CBVSRVUAVHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if (CBVSRVUAVHeapDesc.NumDescriptors > 0)
	{
		debug::check(pDevice->CreateDescriptorHeap(&CBVSRVUAVHeapDesc, IID_PPV_ARGS(&m_CBVSRVUAVDescriptorHeap)));
		m_CBVSRVUAVDescriptorHeap->SetName(L"CbvSrvUavDescriptorHeap");
		m_CBVSRVUAVDescriptorHeapAttributes.init(CBVSRVUAVHeapDesc, m_CBVSRVUAVDescriptorHeap.Get());
	}
	else
	{
		logger::info("CBV_SRV_UAV descriptor heap not created because no descriptor needed.");
	}

	D3D12_DESCRIPTOR_HEAP_DESC SamplerHeapDesc = {};
	SamplerHeapDesc.NumDescriptors = vSamplerNum;
	SamplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
	SamplerHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if (SamplerHeapDesc.NumDescriptors > 0)
	{
		debug::check(pDevice->CreateDescriptorHeap(&SamplerHeapDesc, IID_PPV_ARGS(&m_SamplerDescriptorHeap)));
		m_SamplerDescriptorHeap->SetName(L"SamplerDescriptorHeap");
		m_SamplerDescriptorHeapAttributes.init(SamplerHeapDesc, m_SamplerDescriptorHeap.Get());
	}
	else
	{
		logger::info("Sampler descriptor heap sampler not created because no descriptor needed.");
	}
	
	D3D12_DESCRIPTOR_HEAP_DESC DSVHeapDesc = {};
	DSVHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	DSVHeapDesc.NumDescriptors = vDSVDescriptorNum;
	DSVHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	if (DSVHeapDesc.NumDescriptors > 0)
	{
		debug::check(pDevice->CreateDescriptorHeap(&DSVHeapDesc, IID_PPV_ARGS(&m_DSVDescriptorHeap)));
		m_DSVDescriptorHeap->SetName(L"DSVDescriptorHeap");
		m_DSVDescriptorHeapAttributes.init(DSVHeapDesc, m_DSVDescriptorHeap.Get());
	}
	else
	{
		logger::info("DSV descriptor heap not created because no descriptor needed.");
	}
}

void CApplication::createUnorderedAccessDescriptor(int vIndex, const D3D12_UNORDERED_ACCESS_VIEW_DESC& vDesc, ID3D12Resource* vResource, ID3D12Resource* vCounterResource)
{
	auto pDevice = CIgniter::get()->fetchDevice();
	CD3DX12_CPU_DESCRIPTOR_HANDLE Handle(m_CBVSRVUAVDescriptorHeapAttributes.CPUHandle);
	Handle.Offset(vIndex, m_CBVSRVUAVDescriptorHeapAttributes.DescriptorSize);
	pDevice->CreateUnorderedAccessView(vResource, vCounterResource, &vDesc, Handle);
}

void CApplication::createShaderResourceDescriptor(int vIndex, const D3D12_SHADER_RESOURCE_VIEW_DESC& vDesc, ID3D12Resource* vResource)
{
	auto pDevice = CIgniter::get()->fetchDevice();
	CD3DX12_CPU_DESCRIPTOR_HANDLE Handle(m_CBVSRVUAVDescriptorHeapAttributes.CPUHandle);
	Handle.Offset(vIndex, m_CBVSRVUAVDescriptorHeapAttributes.DescriptorSize);
	pDevice->CreateShaderResourceView(vResource, &vDesc, Handle);
}

void CApplication::createDepthStencilDescriptor(int vIndex, const D3D12_DEPTH_STENCIL_VIEW_DESC& vDesc, ID3D12Resource* vResource)
{
	auto pDevice = CIgniter::get()->fetchDevice();
	CD3DX12_CPU_DESCRIPTOR_HANDLE Handle(m_DSVDescriptorHeapAttributes.CPUHandle);
	Handle.Offset(vIndex, m_DSVDescriptorHeapAttributes.DescriptorSize);
	pDevice->CreateDepthStencilView(vResource, &vDesc, Handle);
}

void CApplication::createShaderResourceDescriptors(std::vector<config::dx::SShaderResource>& vShaderResources)
{
	_ASSERTE(vShaderResources.size() == m_CBVSRVUAVDescriptorHeapAttributes.Desc.NumDescriptors);
	
	auto pDevice = CIgniter::get()->fetchDevice();
	CD3DX12_CPU_DESCRIPTOR_HANDLE Handle(m_CBVSRVUAVDescriptorHeapAttributes.CPUHandle);

	for (auto& ShaderResource : vShaderResources)
	{
		pDevice->CreateShaderResourceView(ShaderResource.pResource, &ShaderResource.Desc, Handle);
		Handle.Offset(1, m_CBVSRVUAVDescriptorHeapAttributes.DescriptorSize);
	}
}

void CApplication::createDepthStencilDescriptors(std::vector<config::dx::SDepthStencil>& vDepthStencils)
{
	_ASSERTE(vDepthStencils.size() == m_DSVDescriptorHeapAttributes.Desc.NumDescriptors);

	auto pDevice = CIgniter::get()->fetchDevice();
	CD3DX12_CPU_DESCRIPTOR_HANDLE Handle(m_DSVDescriptorHeapAttributes.CPUHandle);

	for (auto& DepthStencil : vDepthStencils)
	{
		pDevice->CreateDepthStencilView(DepthStencil.pResource, &DepthStencil.Desc, Handle);
		Handle.Offset(1, m_DSVDescriptorHeapAttributes.DescriptorSize);
	}
}

IDxcBlob* CApplication::compileShaderLibrary(const std::wstring& vFilePath, const std::wstring& vTarget)
{
	// TODO: pointer not released
	debug::check(DxcDllHelper.Initialize());
	
	IDxcCompiler* pCompiler;
	IDxcLibrary* pLibrary;
	debug::check(DxcDllHelper.CreateInstance(CLSID_DxcCompiler, &pCompiler));
	debug::check(DxcDllHelper.CreateInstance(CLSID_DxcLibrary, &pLibrary));

	std::ifstream ShaderFile(vFilePath.c_str());
	std::stringstream StrStream;
	StrStream << ShaderFile.rdbuf();
	std::string Shader = StrStream.str();

	IDxcBlobEncoding* pTextBlob;
	debug::check(pLibrary->CreateBlobWithEncodingFromPinned((LPBYTE)Shader.c_str(), (UINT32)Shader.size(), 0, &pTextBlob));

	IDxcOperationResult* pResult;
	// TODO: other arguments may used in shader too
	debug::check(pCompiler->Compile(pTextBlob, vFilePath.c_str(), L"", vTarget.c_str(), nullptr, 0, nullptr, 0, nullptr, &pResult));
	
	HResult ResultCode;
	debug::check(pResult->GetStatus(&ResultCode));
	if (FAILED(ResultCode))
	{
		IDxcBlobEncoding* pError;
		debug::check(pResult->GetErrorBuffer(&pError));
		std::string log = convertBlobToString(pError);
		logger::error(log);
	}

	IDxcBlob* pBlob;
	debug::check(pResult->GetResult(&pBlob));
	return pBlob;
}


template<class BlotType>
std::string convertBlobToString(BlotType* pBlob)
{
	std::vector<char> infoLog(pBlob->GetBufferSize() + 1);
	memcpy(infoLog.data(), pBlob->GetBufferPointer(), pBlob->GetBufferSize());
	infoLog[pBlob->GetBufferSize()] = 0;
	return std::string(infoLog.data());
}