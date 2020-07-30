#include "Igniter.h"
#include "Application.h"

using namespace Microsoft::WRL;
using namespace config;

CIgniter* CIgniter::m_pIgniter = nullptr;

static LRESULT CALLBACK WndProc(HWND vHwnd, UINT vMsg, WPARAM vWParam, LPARAM vLParam);

CIgniter* CIgniter::start(SHInstance vInstance)
{
	m_pIgniter = new CIgniter(vInstance);
	m_pIgniter->_initialize();
	return m_pIgniter;
}

void CIgniter::run(CApplication* pApp)
{
	auto pIgniter = CIgniter::get();
	_ASSERTE(pIgniter->isInitialized());

	pIgniter->m_pCurrentApp = pApp;

	pApp->start();
	
	// Show Window
	ShowWindow(CIgniter::get()->fetchWindow(), SW_NORMAL);

	// Loop
	SMessage Message = {};
	while (Message.message != WM_QUIT)
	{
		if (PeekMessage(&Message, 0, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&Message);
			DispatchMessage(&Message);
		}
	}

	pApp->shutdown();
}

UINT CIgniter::getDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE vType)
{
	return m_Device->GetDescriptorHandleIncrementSize(vType);
}

std::shared_ptr<CCommandQueue> CIgniter::getCommandQueue(D3D12_COMMAND_LIST_TYPE vType)
{
	switch(vType)
	{
	case D3D12_COMMAND_LIST_TYPE_COPY: return m_CopyCommandQueue;
	case D3D12_COMMAND_LIST_TYPE_COMPUTE: return m_ComputeCommandQueue;
	case D3D12_COMMAND_LIST_TYPE_DIRECT: return m_DirectCommandQueue;
	default: throw std::exception();
	}
}

D3D12_CPU_DESCRIPTOR_HANDLE CIgniter::fetchCurrentRTV() const
{
	UINT CurrentBackBufferIndex = m_SwapChain->GetCurrentBackBufferIndex();
	return CD3DX12_CPU_DESCRIPTOR_HANDLE(m_RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), CurrentBackBufferIndex, m_RTVDescriptorSize);
}

Microsoft::WRL::ComPtr<ID3D12Resource> CIgniter::fetchCurrentBackBuffer() const
{
	UINT CurrentBackBufferIndex = m_SwapChain->GetCurrentBackBufferIndex();
	return m_BackBuffers[CurrentBackBufferIndex];
}

ComPtr<ID3D12DescriptorHeap> CIgniter::createDescriptorHeap(UINT vDescriptorNum, D3D12_DESCRIPTOR_HEAP_TYPE vType)
{
	D3D12_DESCRIPTOR_HEAP_DESC DescriptorHeapDesc = {
		vType,
		vDescriptorNum,
		D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
		0
	};
	ComPtr<ID3D12DescriptorHeap> DescriptorHeap;
	debug::check(m_Device->CreateDescriptorHeap(&DescriptorHeapDesc, IID_PPV_ARGS(&DescriptorHeap)));
	return DescriptorHeap;
}

void CIgniter::processKeyboardInput(WPARAM vParam)
{
	SMessage CharMsg;
	if (PeekMessage(&CharMsg, m_HWindow, 0, 0, PM_NOREMOVE) && CharMsg.message == WM_CHAR)
		GetMessage(&CharMsg, m_HWindow, 0, 0);
	
	m_pCurrentApp->onKey(vParam);

	switch (vParam)
	{
	case VK_ESCAPE: PostQuitMessage(0); break;
	default: break;
	}
}

void CIgniter::_initialize()
{
	__initSystem();
	__initWindow();
	__initDX();
	m_IsInitialized = true;
}

CIgniter::~CIgniter()
{
	m_CopyCommandQueue->Flush();
	m_ComputeCommandQueue->Flush();
	m_DirectCommandQueue->Flush();
	
	for (int i = 0; i < dx::BackBufferCount; i++)
		m_BackBuffers[i].Reset();
}

void CIgniter::__initSystem()
{
	// Redirect standard output
	AllocConsole();
	FILE* fp = nullptr;
	freopen_s(&fp, "CONIN$", "r", stdin);
	freopen_s(&fp, "CONOUT$", "w", stdout);
	freopen_s(&fp, "CONOUT$", "w", stderr);

	// Windows 10 Creators update adds Per Monitor V2 DPI awareness context.
	// Using this awareness context allows the client area of the window to achieve 100% scaling while still allowing non-client window content to be rendered in a DPI sensitive fashion.
	SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
}

void CIgniter::__initWindow()
{
	// Register window
	SWndClassEx WndClass = window::generateWindowClassEx(m_HInstance, WndProc);
	if (!RegisterClassEx(&WndClass))
	{
		prompt::error("Failed to register window, error-" + std::to_string(GetLastError()));
		return;
	}

	// Create Window
	m_HWindow = CreateWindow(window::ClassName.c_str(), window::Title.c_str(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, window::Width, window::Height, NULL, NULL, m_HInstance, NULL);
	if (!m_HWindow)
	{
		prompt::error("Failed to create window, error-" + std::to_string(GetLastError()));
		return;
	}
}

void CIgniter::__initDX()
{
	// Enable debug layer
#ifdef _DEBUG
	ComPtr<ID3D12Debug> DebugInterface;
	debug::check(D3D12GetDebugInterface(IID_PPV_ARGS(&DebugInterface)));
	DebugInterface->EnableDebugLayer();
#endif
	
	// Choose adapter
	ComPtr<IDXGIFactory4> Factory;
	debug::check(CreateDXGIFactory2(0, IID_PPV_ARGS(&Factory)));

	ComPtr<IDXGIAdapter1> Adapter1;
	ComPtr<IDXGIAdapter4> Adapter4;

	if (dx::IsWARPAdapter)
	{
		debug::check(Factory->EnumWarpAdapter(IID_PPV_ARGS(&Adapter1)));
		debug::check(Adapter1.As(&Adapter4));
	}
	else
	{
		SIZE_T MaxMemory = 0;
		for (UINT i = 0; Factory->EnumAdapters1(i, &Adapter1) != DXGI_ERROR_NOT_FOUND; i++)
		{
			DXGI_ADAPTER_DESC1 AdapterDesc;
			Adapter1->GetDesc1(&AdapterDesc);

			if ((AdapterDesc.Flags && DXGI_ADAPTER_FLAG_SOFTWARE) == 0 
				&& SUCCEEDED(D3D12CreateDevice(Adapter1.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)) 
				&& AdapterDesc.DedicatedVideoMemory > MaxMemory)
			{
				MaxMemory = AdapterDesc.DedicatedVideoMemory;
				debug::check(Adapter1.As(&Adapter4));
			}
		}
	}

	// Create device
	debug::check(D3D12CreateDevice(Adapter4.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_Device)));

	// Debug Information
#ifdef _DEBUG
	ID3D12InfoQueue* pInfoQueue = nullptr;
	debug::check(m_Device->QueryInterface(IID_PPV_ARGS(&pInfoQueue)));
	
	pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
	pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
	pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);

	D3D12_MESSAGE_SEVERITY DenySeverities[] =
	{
		D3D12_MESSAGE_SEVERITY_INFO
	};

	D3D12_INFO_QUEUE_FILTER InfoQueueFilter = {};
	InfoQueueFilter.DenyList.NumSeverities = _countof(DenySeverities);
	InfoQueueFilter.DenyList.pSeverityList = DenySeverities;

	debug::check(pInfoQueue->PushStorageFilter(&InfoQueueFilter));
	pInfoQueue->Release();
#endif

	// Create command queue
	m_DirectCommandQueue  = std::make_shared<CCommandQueue>(D3D12_COMMAND_LIST_TYPE_DIRECT);
	m_CopyCommandQueue    = std::make_shared<CCommandQueue>(D3D12_COMMAND_LIST_TYPE_COPY);
	m_ComputeCommandQueue = std::make_shared<CCommandQueue>(D3D12_COMMAND_LIST_TYPE_COMPUTE);

	// Create swap chain
	ComPtr<IDXGISwapChain1> SwapChain;
	DXGI_SWAP_CHAIN_DESC1 SwapChainDesc = dx::generateSwapChainDesc();
	debug::check(Factory->CreateSwapChainForHwnd(m_DirectCommandQueue->fetchCommandQueue().Get(), m_HWindow, &SwapChainDesc, nullptr, nullptr, &SwapChain));
	debug::check(SwapChain.As(&m_SwapChain));

	m_RTVDescriptorHeap = createDescriptorHeap(dx::BackBufferCount, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	// Create RTV Descriptors
	m_RTVDescriptorSize = getDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	CD3DX12_CPU_DESCRIPTOR_HANDLE RTVHandle(m_RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	for (int i = 0; i < dx::BackBufferCount; i++)
	{
		debug::check(m_SwapChain->GetBuffer(i, IID_PPV_ARGS(&(m_BackBuffers[i]))));
		m_Device->CreateRenderTargetView(m_BackBuffers[i].Get(), nullptr, RTVHandle);
		RTVHandle.Offset(m_RTVDescriptorSize);
	}
}

static LRESULT CALLBACK WndProc(HWND vHwnd, UINT vMsg, WPARAM vWParam, LPARAM vLParam)
{
	LRESULT ret = 0;

	switch (vMsg)
	{
	case WM_PAINT:
		CIgniter::get()->m_pCurrentApp->update();
		CIgniter::get()->m_pCurrentApp->render();
		break;
	case WM_KEYDOWN:
		CIgniter::get()->processKeyboardInput(vWParam);
		break;
	case WM_DESTROY:
		PostQuitMessage(ret);
		break;
	default:
		ret = DefWindowProc(vHwnd, vMsg, vWParam, vLParam);
	}

	return ret;
}
