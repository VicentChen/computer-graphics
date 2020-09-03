#include "Buffer.h"
#include "Igniter.h"

using namespace Microsoft::WRL;

void CGPUResource::reset()
{
	logger::info("Reset resource");

	m_pData.Reset();
	m_pIntermediaBuffer.Reset();

	m_Status = FREE;
}

void CDefaultHeapResource::init(CD3DX12_RESOURCE_DESC& vDesc, D3D12_RESOURCE_STATES vInitialState, D3D12_CLEAR_VALUE* vOptimizedClearValue)
{
	debug::check(m_Status == FREE, "Illegal resource status, please reset before reuse");
	
	auto pDevice = CIgniter::get()->fetchDevice();
	debug::check(pDevice->CreateCommittedResource(&global::dx::DefaultHeadProps, D3D12_HEAP_FLAG_NONE, &vDesc, vInitialState, vOptimizedClearValue, IID_PPV_ARGS(&m_pData)));

	m_CurrentState = vInitialState;
	markAsInit();
}

void CDefaultHeapResource::initAsBuffer(ID3D12GraphicsCommandList* vCommandList, const void* vData, size_t vSize, D3D12_RESOURCE_STATES vStateAfterInit, D3D12_CLEAR_VALUE* vOptimizedClearValue)
{
	debug::check(m_Status == FREE, "Illegal resource status, please reset before reuse");
	
	auto pDevice = CIgniter::get()->fetchDevice();
	debug::check(pDevice->CreateCommittedResource(&global::dx::DefaultHeadProps, D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(vSize), D3D12_RESOURCE_STATE_COPY_DEST, vOptimizedClearValue, IID_PPV_ARGS(&m_pData)));
	markAsInit();
	
	copyFrom(vCommandList, vData, vSize);
	m_CurrentState = D3D12_RESOURCE_STATE_COPY_DEST;
	transit(vCommandList, m_CurrentState, vStateAfterInit);
}

void CDefaultHeapResource::copyFrom(ID3D12GraphicsCommandList* vCommandList, const void* vData, size_t vSize)
{
	debug::check(m_Status == INITIALIZED, "Illegal resource status, please initialize before use or reset before reuse");
	
	auto pDevice = CIgniter::get()->fetchDevice();
	debug::check(pDevice->CreateCommittedResource(&global::dx::UploadHeadProps, D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(vSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_pIntermediaBuffer)));
	D3D12_SUBRESOURCE_DATA SubresourceData = { vData, vSize, vSize };
	UpdateSubresources(vCommandList, m_pData.Get(), m_pIntermediaBuffer.Get(), 0, 0, 1, &SubresourceData);
	
	markAsOccupied();
}

void CDefaultHeapResource::transit(ID3D12GraphicsCommandList* vCommandList, D3D12_RESOURCE_STATES vCurrentState, D3D12_RESOURCE_STATES vTargetState)
{
	debug::check(m_Status == INITIALIZED || m_Status == OCCUPIED, "Illegal resource status, please initialize before use or reset before reuse");
	debug::check(vCurrentState == m_CurrentState, "State not match");
	
	CD3DX12_RESOURCE_BARRIER Barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_pData.Get(), vCurrentState, vTargetState);
	vCommandList->ResourceBarrier(1, &Barrier);
	m_CurrentState = vTargetState;
}

void CUploadHeapResource::init(CD3DX12_RESOURCE_DESC& vDesc, D3D12_CLEAR_VALUE* vOptimizedClearValue)
{
	debug::check(m_Status == FREE, "Illegal resource status, please reset before reuse");
	
	auto pDevice = CIgniter::get()->fetchDevice();
	debug::check(pDevice->CreateCommittedResource(&global::dx::UploadHeadProps, D3D12_HEAP_FLAG_NONE, &vDesc, D3D12_RESOURCE_STATE_GENERIC_READ, vOptimizedClearValue, IID_PPV_ARGS(&m_pData)));

	m_CurrentState = D3D12_RESOURCE_STATE_GENERIC_READ;
	markAsInit();
}

void CUploadHeapResource::copyFrom(const void* vData, size_t vSize)
{
	debug::check(m_Status == INITIALIZED || m_Status == OCCUPIED, "Illegal resource status, please initialize before use or reset before reuse");
	
	void* pUnifiedAddr = nullptr;
	m_pData->Map(0, nullptr, &pUnifiedAddr);
	memcpy(pUnifiedAddr, vData, vSize);
	m_pData->Unmap(0, nullptr);

	markAsOccupied();
}
