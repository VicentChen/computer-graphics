#pragma once
#include "Application.h"
#include "RayTracing.h"

struct SShaderTableEntry
{
	UINT64 Identifier[4];
	UINT64 ResourceAddr;
	UINT64 Padding[3];

	SShaderTableEntry(void* vIdentifier, UINT64 vResourceAddr) : ResourceAddr(vResourceAddr)
	{
		memcpy(Identifier, vIdentifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
		for (int i = 0; i < 3; i++) Padding[i] = 0;
	}
};

class CRayTracingTriangle : public CApplication
{
	CDefaultHeapResource m_VertexBuffer[2];
	CDefaultHeapResource m_pOutput;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_pColor[3];

	D3D12_VERTEX_BUFFER_VIEW m_VertexBufferView[2];

	Microsoft::WRL::ComPtr<ID3D12Resource> m_pTriangleBLAS;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_pPlaneBLAS;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_pTLAS;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_pShaderTable;

	Microsoft::WRL::ComPtr<ID3D12RootSignature> m_pRayGenRootSignature;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> m_pHitRootSignature;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> m_pMissRootSignature;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> m_pGlobalRootSignature;

	Microsoft::WRL::ComPtr<ID3D12StateObject> m_pPipelineState;
	
	CShaderTable<SShaderTableEntry> m_ShaderTable;
	
	UINT32 m_ShaderTableEntrySize = 0;
	
public:
	void render() override;
	
protected:
	void _initPipeline() override;
	void _loadModels() override;
	void _describeAssets() override;
};