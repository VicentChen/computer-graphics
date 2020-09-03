#include "CommandQueue.h"
#include "Igniter.h"

using namespace Microsoft::WRL;

CCommandQueue::CCommandQueue(D3D12_COMMAND_LIST_TYPE vType) : m_CommandListType(vType)
{
	D3D12_COMMAND_QUEUE_DESC CommandQueueDesc = {
		vType,
		D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
		D3D12_COMMAND_QUEUE_FLAG_NONE,
		0
	};

	auto Device = CIgniter::get()->fetchDevice();
	
	debug::check(Device->CreateCommandQueue(&CommandQueueDesc, IID_PPV_ARGS(&m_CommandQueue)));
	debug::check(Device->CreateFence(m_FenceValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_Fence)));
	
	m_FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	if (!m_FenceEvent) logger::error("Failed to create fence event");
}

ComPtr<ID3D12GraphicsCommandList4> CCommandQueue::createCommandList()
{
	auto Device = CIgniter::get()->fetchDevice();
	
	ComPtr<ID3D12CommandAllocator> Allocator;
	//debug::check(Device->CreateCommandAllocator(m_CommandListType, IID_PPV_ARGS(&Allocator)));
	if (FAILED(Device->CreateCommandAllocator(m_CommandListType, IID_PPV_ARGS(&Allocator))))
	{
		auto I = CIgniter::get()->fetchDevice()->GetDeviceRemovedReason();
			debug::check(I);
	}

	ComPtr<ID3D12GraphicsCommandList4> CommandList;
	//debug::check(Device->CreateCommandList(0, m_CommandListType, Allocator.Get(), nullptr, IID_PPV_ARGS(&CommandList)));
	if (FAILED(Device->CreateCommandList(0, m_CommandListType, Allocator.Get(), nullptr, IID_PPV_ARGS(&CommandList))))
	{
		auto I = CIgniter::get()->fetchDevice()->GetDeviceRemovedReason();
		debug::check(I);
	}
	return CommandList;
}

UINT64 CCommandQueue::executeCommandList(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> vCommandList)
{
	vCommandList->Close();
	ID3D12CommandList* const ppCommandLists[] = {
		vCommandList.Get()
	};

	m_CommandQueue->ExecuteCommandLists(1, ppCommandLists);
	
	return signal();
}

UINT64 CCommandQueue::signal()
{
	UINT64 FenceValue = ++m_FenceValue;
	m_CommandQueue->Signal(m_Fence.Get(), FenceValue);
	return FenceValue;
}

bool CCommandQueue::isFenceComplete(UINT64 vValue) const
{
	return m_Fence->GetCompletedValue() >= vValue;
}

void CCommandQueue::wait4Fence(UINT64 vValue)
{
	if (!isFenceComplete(vValue))
	{
		debug::check(m_Fence->SetEventOnCompletion(vValue, m_FenceEvent));
		WaitForSingleObject(m_FenceEvent, DWORD_MAX);
	}
}

void CCommandQueue::Flush()
{
	wait4Fence(signal());
}

