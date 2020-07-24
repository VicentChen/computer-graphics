#define _CRT_SECURE_NO_WARNINGS
#define _DISABLE_EXTENDED_ALIGNED_STORAGE 

#include "Helpers.h"
#include "Application.h"
#include "Game.h"
#include "CommandQueue.h"
#include "Window.h"
#include "Tutorial2.h"

#include <iostream>
#include <d3dx12.h>

class CillyGame : public Game
{
public:
	CillyGame(const std::wstring& name, int width, int height, bool vSync) : Game(name, width, height, vSync) {}
	bool LoadContent() override { return true; }
	void UnloadContent() override { }

	void OnRender(RenderEventArgs& e) override
	{
		auto commandQueue = Application::Get().GetCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
		auto commandList = commandQueue->GetCommandList();

		UINT currentBackBufferIndex = m_pWindow->GetCurrentBackBufferIndex();
		auto backBuffer = m_pWindow->GetCurrentBackBuffer();
		auto rtv = m_pWindow->GetCurrentRenderTargetView();

		// Clear the render targets.
		{
			CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
			commandList->ResourceBarrier(1, &barrier);

			FLOAT clearColor[] = { 0.4f, 0.6f, 0.9f, 1.0f };

			auto RTV = m_pWindow->GetCurrentRenderTargetView();
			commandList->ClearRenderTargetView(RTV, clearColor, 0, nullptr);
		}

		// present
		{
			CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
			commandList->ResourceBarrier(1, &barrier);
			
			m_FenceValues[currentBackBufferIndex] = commandQueue->ExecuteCommandList(commandList);

			currentBackBufferIndex = m_pWindow->Present();

			commandQueue->WaitForFenceValue(m_FenceValues[currentBackBufferIndex]);
		}
	}

private:
	uint64_t m_FenceValues[Window::BufferCount] = {};
};

int CALLBACK wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow)
{
	//AllocConsole();
	//freopen("conin$", "r", stdin);
	//freopen("conout$", "w", stdout);
	//freopen("conout$", "w", stderr);
	
	//std::shared_ptr<CillyGame> pGame = std::make_shared<CillyGame>(L"Learn DX 12", 1920, 1080, false);
	std::shared_ptr<Tutorial2> pGame = std::make_shared<Tutorial2>(L"Learn DX 12", 1920, 1080, false);
	
	Application::Create(hInstance);

	Application& application = Application::Get();

	Application::Get().Run(pGame);
	
	Application::Destroy();

	return 0;
}

