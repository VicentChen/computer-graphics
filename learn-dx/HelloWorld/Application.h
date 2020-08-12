#pragma once
#include "Common.h"

class CApplication
{
	struct SDescriptorHeapAttributes
	{
		int DescriptorSize;
		D3D12_DESCRIPTOR_HEAP_DESC Desc;
		D3D12_CPU_DESCRIPTOR_HANDLE CPUHandle;
		D3D12_GPU_DESCRIPTOR_HANDLE GPUHandle;

		void init(D3D12_DESCRIPTOR_HEAP_DESC& vDesc, ID3D12DescriptorHeap* vHeap);
	};
	
protected:
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_CBVSRVUAVDescriptorHeap;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_SamplerDescriptorHeap;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_DSVDescriptorHeap;

	SDescriptorHeapAttributes m_CBVSRVUAVDescriptorHeapAttributes;
	SDescriptorHeapAttributes m_SamplerDescriptorHeapAttributes;
	SDescriptorHeapAttributes m_DSVDescriptorHeapAttributes;
	
public:

	CApplication() = default;
	virtual ~CApplication() = default;

	virtual void start();
	virtual void update();
	virtual void render();
	virtual void shutdown() {}

	virtual void onKey(WPARAM vParam) {}

	void createDescriptorHeap(int vCBVDescriptorCount, int vSRVDescriptorNum, int vUAVDescriptorNum, int vSamplerNum, int vDSVDescriptorNum);
	void createShaderResourceDescriptor(int vIndex, const D3D12_SHADER_RESOURCE_VIEW_DESC& vDesc, ID3D12Resource* vResource);
	void createDepthStencilDescriptor(int vIndex, const D3D12_DEPTH_STENCIL_VIEW_DESC& vDesc, ID3D12Resource* vResource);
};
