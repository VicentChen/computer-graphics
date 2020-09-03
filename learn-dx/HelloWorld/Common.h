#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <wrl.h>

#include "d3dx12.h"
#include <dxgi1_6.h>
#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <dxcapi.h>

#include <array>
#include <string>
#include <iostream>

#define NUM_ARRAY_ARGS(Array) _countof(Array), Array

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

	inline void check(bool b, const std::string& vInfo = "No error message")
	{
		if (!b)
		{
			logger::error(vInfo);
			throw std::exception();
		}
	}
}

namespace config
{
	namespace window
	{
		extern const std::string ClassName;
		extern const std::string Title;
		
		extern const int Width;
		extern const int Height;

		extern float ClearColor[];
		
		SWndClassEx generateWindowClassEx(SHInstance vHInstance, WNDPROC vWndProc);
	}

	namespace dx
	{	
		struct SShaderResource
		{
			ID3D12Resource* pResource;
			D3D12_SHADER_RESOURCE_VIEW_DESC Desc;
		};

		struct SDepthStencil
		{
			ID3D12Resource* pResource;
			D3D12_DEPTH_STENCIL_VIEW_DESC Desc;
		};
		
		extern const bool IsWARPAdapter;
		extern const UINT BackBufferCount;
		
		DXGI_SWAP_CHAIN_DESC1 generateSwapChainDesc();
	}

	namespace shader
	{
		extern const std::string EntryPoint;
		extern const std::string VSTarget;
		extern const std::string PSTarget;
	}
}

namespace global
{
	namespace dx
	{
		extern const CD3DX12_HEAP_PROPERTIES DefaultHeadProps;
		extern const CD3DX12_HEAP_PROPERTIES UploadHeadProps;
	}
}
