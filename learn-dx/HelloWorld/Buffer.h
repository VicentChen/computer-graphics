#pragma once
#include "Common.h"

class CGPUResource
{
protected:
	enum EStatus { FREE, INITIALIZED, OCCUPIED };

	CGPUResource() = default;
	virtual ~CGPUResource() = default;
	
	D3D12_RESOURCE_STATES m_CurrentState = D3D12_RESOURCE_STATE_COMMON;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_pData;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_pIntermediaBuffer;

	EStatus m_Status = FREE;
	
public:
	D3D12_GPU_VIRTUAL_ADDRESS getGPUVirtualAddress() const { return m_pData->GetGPUVirtualAddress(); }
	ID3D12Resource* fetchResourcePtr() const { return m_pData.Get(); }	

	void reset();
	
protected:
	void markAsInit() { m_Status = INITIALIZED; }
	void markAsOccupied() { m_Status = OCCUPIED; }
};

class CDefaultHeapResource : public CGPUResource
{
public:
	CDefaultHeapResource() = default;
	~CDefaultHeapResource() = default;

	void init(CD3DX12_RESOURCE_DESC& vDesc, D3D12_RESOURCE_STATES vInitialState, D3D12_CLEAR_VALUE* vOptimizedClearValue = nullptr);
	void initAsBuffer(ID3D12GraphicsCommandList* vCommandList, const void* vData, size_t vSize, D3D12_RESOURCE_STATES vStateAfterInit, D3D12_CLEAR_VALUE* vOptimizedClearValue = nullptr);
	void copyFrom(ID3D12GraphicsCommandList* vCommandList, const void* vData, size_t vSize);
	void transit(ID3D12GraphicsCommandList* vCommandList, D3D12_RESOURCE_STATES vCurrentState, D3D12_RESOURCE_STATES vTargetState);
};

class CUploadHeapResource : public CGPUResource
{
public:
	CUploadHeapResource() = default;
	~CUploadHeapResource() = default;

	void init(CD3DX12_RESOURCE_DESC& vDesc, D3D12_CLEAR_VALUE* vOptimizedClearValue = nullptr);
	void copyFrom(const void* vData, size_t vSize);
};