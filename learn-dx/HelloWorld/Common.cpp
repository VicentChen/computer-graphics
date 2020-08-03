#include "Common.h"

namespace config
{
	namespace window
	{
		const std::string ClassName = "Learn DX 12";
		const std::string Title = "Learn DX 12";

		const int Width = 1920;
		const int Height = 1080;

		float ClearColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };

		SWndClassEx generateWindowClassEx(SHInstance vHInstance, WNDPROC vWndProc);
	}

	namespace dx
	{
		const bool IsWARPAdapter = false;
		const UINT BackBufferCount = 3;
	}

	namespace shader
	{
		const std::string EntryPoint = "main";
		const std::string VSTarget = "vs_5_1";
		const std::string PSTarget = "ps_5_1";
	}
}

SWndClassEx config::window::generateWindowClassEx(SHInstance vHInstance, WNDPROC vWndProc)
{
	SWndClassEx ClassEx = {
		sizeof(SWndClassEx),
		CS_HREDRAW | CS_VREDRAW,
		vWndProc,
		0,
		0,
		vHInstance,
		LoadIcon(nullptr, IDI_APPLICATION),
		LoadCursor(nullptr, IDC_ARROW),
		(HBRUSH)GetStockObject(GRAY_BRUSH),
		nullptr,
		ClassName.c_str(),
		LoadIcon(nullptr, IDI_APPLICATION)
	};
	return ClassEx;
}

DXGI_SWAP_CHAIN_DESC1 config::dx::generateSwapChainDesc()
{
	DXGI_SWAP_CHAIN_DESC1 SwapChainDesc;
	SwapChainDesc.Width = config::window::Width;
	SwapChainDesc.Height = config::window::Height;
	SwapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	SwapChainDesc.Stereo = FALSE;
	SwapChainDesc.SampleDesc = { 1, 0 };
	SwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	SwapChainDesc.BufferCount = BackBufferCount;
	SwapChainDesc.Scaling = DXGI_SCALING_STRETCH;
	SwapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	SwapChainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	SwapChainDesc.Flags = 0;
	
	return SwapChainDesc;
}
