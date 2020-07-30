#include "Common.h"

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
