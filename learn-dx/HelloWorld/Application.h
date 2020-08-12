#pragma once
#include "Common.h"

class CApplication
{
protected:
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_CBVSRVUAVDescriptorHeap;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_SamplerDescriptorHeap;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_DSVDescriptorHeap;
	
public:

	CApplication() = default;
	virtual ~CApplication() = default;

	virtual void start();
	virtual void update();
	virtual void render();
	virtual void shutdown() {}

	virtual void onKey(WPARAM vParam) {}

	void createDescriptorHeap(int vCBVDescriptorCount, int vSRVDescriptorNum, int vUAVDescriptorNum, int vSamplerNum, int vDSVDescriptorNum);
	
};
