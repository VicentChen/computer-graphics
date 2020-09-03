#pragma once
#include "Common.h"

class CCommandQueue
{
	Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_CommandQueue;
	Microsoft::WRL::ComPtr<ID3D12Fence> m_Fence;
	D3D12_COMMAND_LIST_TYPE m_CommandListType;
	Handle m_FenceEvent;
	UINT64 m_FenceValue = 0;

public:
	CCommandQueue(D3D12_COMMAND_LIST_TYPE vType);
	~CCommandQueue() = default;

	Microsoft::WRL::ComPtr<ID3D12CommandQueue> fetchCommandQueue() const { return m_CommandQueue; }

	// TODO: 当前每次都需要创建一个CommandList, 可以使用Pool进行优化
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> createCommandList();
	UINT64 executeCommandList(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> vCommandList);
	
	UINT64 signal();
	bool isFenceComplete(UINT64 vValue) const;
	void wait4Fence(UINT64 vValue);
	void Flush();
};
