#pragma once
#include "CommandQueue.h"

class CApplication;

class CIgniter
{
	friend static LRESULT CALLBACK WndProc(HWND vHwnd, UINT vMsg, WPARAM vWParam, LPARAM vLParam);
	
	static CIgniter* m_pIgniter;

	SHInstance m_HInstance;
	SHWnd m_HWindow;
	Microsoft::WRL::ComPtr<ID3D12Device5> m_Device;

	std::shared_ptr<CCommandQueue> m_DirectCommandQueue;
	std::shared_ptr<CCommandQueue> m_ComputeCommandQueue;
	std::shared_ptr<CCommandQueue> m_CopyCommandQueue;

	Microsoft::WRL::ComPtr<IDXGISwapChain4> m_SwapChain;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_RTVDescriptorHeap;
	
	Microsoft::WRL::ComPtr<ID3D12Resource>* m_pBackBuffers;
	UINT m_RTVDescriptorSize;
	
	bool m_IsInitialized = false;

	CApplication* m_pCurrentApp;

public:
	static CIgniter* start(SHInstance vInstance);
	static void run(CApplication* pApp);
	static void shutdown() { delete m_pIgniter; }
	static CIgniter* get() { _ASSERTE(m_pIgniter); return m_pIgniter; }
	
	bool isInitialized() const { return m_IsInitialized; }
	UINT getDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE vType);
	std::shared_ptr<CCommandQueue> fetchCommandQueue(D3D12_COMMAND_LIST_TYPE vType) const;
	D3D12_CPU_DESCRIPTOR_HANDLE fetchCurrentRTV() const;
	Microsoft::WRL::ComPtr<ID3D12Resource> fetchCurrentBackBuffer() const;
	Microsoft::WRL::ComPtr<ID3D12Device5> fetchDevice() const { return m_Device; }
	Microsoft::WRL::ComPtr<IDXGISwapChain4> fetchSwapChain() const { return m_SwapChain; }
	SHWnd fetchWindow() const { return m_HWindow; }
	
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> createDescriptorHeap(UINT vDescriptorNum, D3D12_DESCRIPTOR_HEAP_TYPE vType);

	void processKeyboardInput(WPARAM vParam);
	
protected:
	void _initialize();

private:
	CIgniter(SHInstance vInstance);
	~CIgniter();

	void __initSystem();
	void __initWindow();
	void __initDX();
};
