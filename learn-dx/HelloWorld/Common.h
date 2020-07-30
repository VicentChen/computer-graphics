#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <wrl.h>

#include "d3dx12.h"
#include <dxgi1_6.h>

#include <string>
#include <iostream>

using SHInstance = HINSTANCE;
using SWndClassEx = WNDCLASSEX;
using SHWnd = HWND;
using SMessage = MSG;
using Handle = HANDLE;
using HResult = HRESULT;

namespace logger
{
	inline void info(const std::string& vMessage)    { std::cout << "[INFO ]: " << vMessage << std::endl; }
	inline void warning(const std::string& vMessage) { std::cout << "[WARN ]: " << vMessage << std::endl; }
	inline void error(const std::string& vMessage)   { std::cout << "[ERROR]: " << vMessage << std::endl; }
}

namespace prompt
{
	inline void info   (const std::string& vMessage, const std::string& vCaption = "Info"   ) { MessageBox(NULL, vMessage.c_str(), vCaption.c_str(), MB_ICONINFORMATION | MB_OK); }
	inline void warning(const std::string& vMessage, const std::string& vCaption = "Warning") { MessageBox(NULL, vMessage.c_str(), vCaption.c_str(), MB_ICONWARNING     | MB_OK); }
	inline void error  (const std::string& vMessage, const std::string& vCaption = "Error"  ) { MessageBox(NULL, vMessage.c_str(), vCaption.c_str(), MB_ICONERROR       | MB_OK); }
}

namespace debug
{
	inline void check(HResult vResult)
	{
		if (FAILED(vResult))
		{
			logger::error("Code - " + std::to_string(vResult));
			throw std::exception();
		}
	}
}

namespace config
{
	namespace window
	{
		const std::string ClassName = "Learn DX 12";
		const std::string Title     = "Learn DX 12";
		
		const int Width  = 1920;
		const int Height = 1080;

		SWndClassEx generateWindowClassEx(SHInstance vHInstance, WNDPROC vWndProc);
	}

	namespace dx
	{
		const bool IsWARPAdapter = false;
		const UINT BackBufferCount = 3;
		
		DXGI_SWAP_CHAIN_DESC1 generateSwapChainDesc();
	}
}