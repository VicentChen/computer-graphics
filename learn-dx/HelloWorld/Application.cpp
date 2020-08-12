#include "Igniter.h"
#include "Application.h"


void CApplication::start()
{
	createDescriptorHeap(0, 0, 0, 0, 1);
}

void CApplication::update()
{
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

void CApplication::createDescriptorHeap(int vCBVDescriptorCount, int vSRVDescriptorNum, int vUAVDescriptorNum, int vSamplerNum, int vDSVDescriptorNum) {
	auto pDevice = CIgniter::get()->fetchDevice();
	
	D3D12_DESCRIPTOR_HEAP_DESC CBVSRVUAVHeapDesc = {};
	CBVSRVUAVHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	CBVSRVUAVHeapDesc.NumDescriptors = vCBVDescriptorCount + vSRVDescriptorNum + vUAVDescriptorNum;
	CBVSRVUAVHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if (CBVSRVUAVHeapDesc.NumDescriptors > 0)
		debug::check(pDevice->CreateDescriptorHeap(&CBVSRVUAVHeapDesc, IID_PPV_ARGS(&m_CBVSRVUAVDescriptorHeap)));
	else
		logger::info("CBV_SRV_UAV descriptor heap not created because no descriptor needed.");

	D3D12_DESCRIPTOR_HEAP_DESC SamplerHeapDesc = {};
	SamplerHeapDesc.NumDescriptors = vSamplerNum;
	SamplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
	SamplerHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if (SamplerHeapDesc.NumDescriptors > 0)
		debug::check(pDevice->CreateDescriptorHeap(&SamplerHeapDesc, IID_PPV_ARGS(&m_SamplerDescriptorHeap)));
	else
		logger::info("Sampler descriptor heap sampler not created because no descriptor needed.");
	
	D3D12_DESCRIPTOR_HEAP_DESC DSVHeapDesc = {};
	DSVHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	DSVHeapDesc.NumDescriptors = vDSVDescriptorNum;
	DSVHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	if (DSVHeapDesc.NumDescriptors > 0)
		debug::check(pDevice->CreateDescriptorHeap(&DSVHeapDesc, IID_PPV_ARGS(&m_DSVDescriptorHeap)));
	else
		logger::info("DSV descriptor heap not created because no descriptor needed.");
}
