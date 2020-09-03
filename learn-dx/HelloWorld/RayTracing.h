#pragma once
#include "Common.h"
#include "Buffer.h"

template <typename TShaderTableEntry>
class CShaderTable
{
	std::vector<TShaderTableEntry> m_ShaderTableEntries;
	CUploadHeapResource m_ShaderTable;
	
public:
	CShaderTable() = default;
	~CShaderTable() = default;

	D3D12_GPU_VIRTUAL_ADDRESS getGPUVirtualAddress() const { return m_ShaderTable.getGPUVirtualAddress(); }
	void addEntry(const TShaderTableEntry& vEntry) { m_ShaderTableEntries.emplace_back(vEntry); }
	void setShaderTable(const std::vector<TShaderTableEntry>& vEntries) { m_ShaderTableEntries = vEntries; }
	
	void uploadShaderTable()
	{
		debug::check(sizeof(TShaderTableEntry) % D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES == 0, "Illegal shader table entry size");
		CD3DX12_RESOURCE_DESC ShaderTableBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(TShaderTableEntry) * m_ShaderTableEntries.size());
		m_ShaderTable.init(ShaderTableBufferDesc);
		m_ShaderTable.copyFrom(m_ShaderTableEntries.data(), sizeof(TShaderTableEntry) * m_ShaderTableEntries.size());
	}
};