#include "RayTracingTriangle.h"
#include "Igniter.h"
#include "dxcapi.use.h"

#define align_to(_alignment, _val) (((_val + _alignment - 1) / _alignment) * _alignment)

using namespace Microsoft::WRL;
using namespace config;
using namespace DirectX;

struct SVertex
{
	XMFLOAT3 Position;
};

struct SAccelerationStructureBuffers
{
	ComPtr<ID3D12Resource> pScratch;
	ComPtr<ID3D12Resource> pResult;
	ComPtr<ID3D12Resource> pInstanceDesc;
};

static SVertex TriangleVertices[3] = {
	{XMFLOAT3(0     ,  1   , 0)},
	{XMFLOAT3(0.866f, -0.5f, 0)},
	{XMFLOAT3(-0.866f, -0.5f, 0)},
};

static SVertex PlaneVertices[6] = {
	{ XMFLOAT3(-100, -1,  -2) },
	{ XMFLOAT3(100, -1,  100) },
	{ XMFLOAT3(-100, -1,  100) },
	{ XMFLOAT3(-100, -1,  -2) },
	{ XMFLOAT3(100, -1,  -2) },
	{ XMFLOAT3(100, -1,  100) },
};

XMFLOAT4 ColorData[] =
{
	// Instance 0
	XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f),
	XMFLOAT4(1.0f, 1.0f, 0.0f, 1.0f),
	XMFLOAT4(1.0f, 0.0f, 1.0f, 1.0f),

	// Instance 1
	XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f),
	XMFLOAT4(0.0f, 1.0f, 1.0f, 1.0f),
	XMFLOAT4(1.0f, 1.0f, 0.0f, 1.0f),

	// Instance 2
	XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f),
	XMFLOAT4(1.0f, 0.0f, 1.0f, 1.0f),
	XMFLOAT4(0.0f, 1.0f, 1.0f, 1.0f),
};

static const std::wstring RayGenShaderEntryPoint = L"rayGen";
static const std::wstring MissShaderEntryPoint = L"miss";
static const std::wstring ClosestHitShaderEntryPoint = L"chs";
static const std::wstring HitGroupEntryPoint = L"HitGroup";
static const std::wstring PlaneCHS = L"planeChs";
static const std::wstring PlaneHitGroupEntryPoint = L"PlaneHitGroup";

void CRayTracingTriangle::render()
{
	auto pIgniter = CIgniter::get();
	auto pCommandQueue = pIgniter->fetchCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto SwapChain = pIgniter->fetchSwapChain();
	auto CommandList = pCommandQueue->createCommandList();
	
	auto RTV = pIgniter->fetchCurrentRTV();
	auto BackBuffer = pIgniter->fetchCurrentBackBuffer();

	m_pOutput.transit(CommandList.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	D3D12_DISPATCH_RAYS_DESC RayTraceDesc = {};
	RayTraceDesc.Width = config::window::Width;
	RayTraceDesc.Height = config::window::Height;
	RayTraceDesc.Depth = 1;
	
	RayTraceDesc.RayGenerationShaderRecord.StartAddress = m_ShaderTable.getGPUVirtualAddress() + 0 * m_ShaderTableEntrySize;
	RayTraceDesc.RayGenerationShaderRecord.SizeInBytes = m_ShaderTableEntrySize;

	RayTraceDesc.MissShaderTable.StartAddress = m_ShaderTable.getGPUVirtualAddress() + 1 * m_ShaderTableEntrySize;
	RayTraceDesc.MissShaderTable.SizeInBytes = m_ShaderTableEntrySize;
	RayTraceDesc.MissShaderTable.StrideInBytes = m_ShaderTableEntrySize;

	RayTraceDesc.HitGroupTable.StartAddress = m_ShaderTable.getGPUVirtualAddress() + 2 * m_ShaderTableEntrySize;
	RayTraceDesc.HitGroupTable.SizeInBytes = m_ShaderTableEntrySize * 4;
	RayTraceDesc.HitGroupTable.StrideInBytes = m_ShaderTableEntrySize;
	
	ID3D12DescriptorHeap* Heaps[] = { m_CBVSRVUAVDescriptorHeap.Get() };

	CommandList->SetDescriptorHeaps(1, Heaps);
	CommandList->SetComputeRootSignature(m_pGlobalRootSignature.Get());
	CommandList->SetPipelineState1(m_pPipelineState.Get());
	CommandList->DispatchRays(&RayTraceDesc);

	m_pOutput.transit(CommandList.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);

	{
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(BackBuffer.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
		CommandList->ResourceBarrier(1, &barrier);
		CommandList->CopyResource(BackBuffer.Get(), m_pOutput.fetchResourcePtr());
	}

	// present
	{
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(BackBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
		CommandList->ResourceBarrier(1, &barrier);
		UINT64 FenceValue = pCommandQueue->executeCommandList(CommandList);
		debug::check(SwapChain->Present(0, 0));
		pCommandQueue->wait4Fence(FenceValue);
	}
}

void CRayTracingTriangle::_initPipeline()
{
	auto pDevice = CIgniter::get()->fetchDevice();
	
	D3D12_STATE_SUBOBJECT Subobjects[13];
	int Index = 0;

	// 0 Library
	IDxcBlob* pDxilLib = compileShaderLibrary(L"RayTracingTriangle.hlsl", L"lib_6_3");
	std::wstring EntryPoints[] = { RayGenShaderEntryPoint, MissShaderEntryPoint, ClosestHitShaderEntryPoint, PlaneCHS };
	D3D12_EXPORT_DESC DxilLibraryExportDescs[] = {
		{ RayGenShaderEntryPoint.c_str(), nullptr, D3D12_EXPORT_FLAG_NONE },
		{ MissShaderEntryPoint.c_str(), nullptr, D3D12_EXPORT_FLAG_NONE },
		{ ClosestHitShaderEntryPoint.c_str(), nullptr, D3D12_EXPORT_FLAG_NONE },
		{ PlaneCHS.c_str(), nullptr, D3D12_EXPORT_FLAG_NONE },
	};
	
	D3D12_DXIL_LIBRARY_DESC DxilLibraryDesc = {};
	DxilLibraryDesc.DXILLibrary.pShaderBytecode = pDxilLib->GetBufferPointer();
	DxilLibraryDesc.DXILLibrary.BytecodeLength = pDxilLib->GetBufferSize();
	DxilLibraryDesc.NumExports = _countof(EntryPoints);
	DxilLibraryDesc.pExports = DxilLibraryExportDescs;

	D3D12_STATE_SUBOBJECT DxilLibSubobject = { D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &DxilLibraryDesc };
	Subobjects[Index++] = DxilLibSubobject;

	// 1 Hit Group
	D3D12_HIT_GROUP_DESC HitGroupDesc = { };
	HitGroupDesc.AnyHitShaderImport = nullptr;
	HitGroupDesc.ClosestHitShaderImport = ClosestHitShaderEntryPoint.c_str();
	HitGroupDesc.HitGroupExport = HitGroupEntryPoint.c_str();
	
	D3D12_STATE_SUBOBJECT HitGroupSubobject = { D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &HitGroupDesc };
	Subobjects[Index++] = HitGroupSubobject;

	// 2 Plane Hit Group
	D3D12_HIT_GROUP_DESC PlaneHitGroupDesc = {};
	PlaneHitGroupDesc.AnyHitShaderImport = nullptr;
	PlaneHitGroupDesc.ClosestHitShaderImport = PlaneCHS.c_str();
	PlaneHitGroupDesc.HitGroupExport = PlaneHitGroupEntryPoint.c_str();

	D3D12_STATE_SUBOBJECT PlaneHitGroupSubobject = { D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &PlaneHitGroupDesc };
	Subobjects[Index++] = PlaneHitGroupSubobject;

	// 3 RayGen Root Signature
	ComPtr<ID3DBlob> pRayGenSignatureBlob, pRayGenErrorBlob;
	
	CD3DX12_DESCRIPTOR_RANGE RayGenDescriptorRanges[2];
	RayGenDescriptorRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0);
	RayGenDescriptorRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 1);
	
	CD3DX12_ROOT_PARAMETER RayGenRootParameters[1];
	RayGenRootParameters[0].InitAsDescriptorTable(2, RayGenDescriptorRanges, D3D12_SHADER_VISIBILITY_ALL);

	CD3DX12_ROOT_SIGNATURE_DESC RayGenRootSignatureDesc(NUM_ARRAY_ARGS(RayGenRootParameters), 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);
	debug::check(D3D12SerializeRootSignature(&RayGenRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &pRayGenSignatureBlob, &pRayGenErrorBlob));
	debug::check(pDevice->CreateRootSignature(0, pRayGenSignatureBlob->GetBufferPointer(), pRayGenSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&m_pRayGenRootSignature)));
	ID3D12RootSignature* pRayGenRootSignature = m_pRayGenRootSignature.Get();
	D3D12_STATE_SUBOBJECT RayGenRootSignatureSubobject = { D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE, &pRayGenRootSignature };
	Subobjects[Index++] = RayGenRootSignatureSubobject;

	// 4 Associate Root Signature to RayGen Signature
	auto pRayGenShaderEntryPoint = RayGenShaderEntryPoint.c_str();
	D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION RayGenAssociation = {};
	RayGenAssociation.NumExports = 1;
	RayGenAssociation.pExports = &pRayGenShaderEntryPoint;
	RayGenAssociation.pSubobjectToAssociate = &(Subobjects[Index - 1]);

	D3D12_STATE_SUBOBJECT RayGenAssociationSubobject = { D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION, &RayGenAssociation };
	Subobjects[Index++] = RayGenAssociationSubobject;

	// 5 Hit Root
	ComPtr<ID3DBlob> pHitSignatureBlob, pHitErrorBlob;

	CD3DX12_ROOT_PARAMETER HitRootParameter;
	HitRootParameter.InitAsConstantBufferView(0);
	CD3DX12_ROOT_SIGNATURE_DESC HitRootSignatureDesc(1, &HitRootParameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);
	debug::check(D3D12SerializeRootSignature(&HitRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &pHitSignatureBlob, &pHitErrorBlob));
	debug::check(pDevice->CreateRootSignature(0, pHitSignatureBlob->GetBufferPointer(), pHitSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&m_pHitRootSignature)));
	ID3D12RootSignature* pHitRootSignature = m_pHitRootSignature.Get();
	D3D12_STATE_SUBOBJECT HitRootSignatureSubobject = { D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE, &pHitRootSignature };
	Subobjects[Index++] = HitRootSignatureSubobject;

	// 6
	auto pHitShaderEntryPoint = ClosestHitShaderEntryPoint.c_str();
	D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION HitAssociation = {};
	HitAssociation.NumExports = 1;
	HitAssociation.pExports = &pHitShaderEntryPoint;
	HitAssociation.pSubobjectToAssociate = &(Subobjects[Index - 1]);
	D3D12_STATE_SUBOBJECT HitAssociationSubobject = { D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION, &HitAssociation };
	Subobjects[Index++] = HitAssociationSubobject;
	
	// 7
	ComPtr<ID3DBlob> pMissSignatureBlob, pMissErrorBlob;
	D3D12_ROOT_SIGNATURE_DESC MissRootSignatureDesc = {};
	MissRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
	debug::check(D3D12SerializeRootSignature(&MissRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &pMissSignatureBlob, &pMissErrorBlob));
	debug::check(pDevice->CreateRootSignature(0, pMissSignatureBlob->GetBufferPointer(), pMissSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&m_pMissRootSignature)));
	ID3D12RootSignature* pMissRootSignature = m_pMissRootSignature.Get();
	D3D12_STATE_SUBOBJECT MissRootSignatureSubobject = { D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE, &pMissRootSignature };
	Subobjects[Index++] = MissRootSignatureSubobject;
	
	// 8
	const WCHAR* MissShaderEntryPoints[] = { MissShaderEntryPoint.c_str(), PlaneCHS.c_str() };
	D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION MissAssociation = {};
	MissAssociation.NumExports = 2;
	MissAssociation.pExports = MissShaderEntryPoints;
	MissAssociation.pSubobjectToAssociate = &(Subobjects[Index - 1]);
	D3D12_STATE_SUBOBJECT MissAssociationSubobject = { D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION, &MissAssociation };
	Subobjects[Index++] = MissAssociationSubobject;

	// 9 Shader config
	D3D12_RAYTRACING_SHADER_CONFIG ShaderConfig = { sizeof(float) * 3, sizeof(float) * 2 };

	D3D12_STATE_SUBOBJECT ShaderConfigSubobject = { D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &ShaderConfig };
	Subobjects[Index++] = ShaderConfigSubobject;

	// 10 Associate Shader Config to Miss, CHS, RGS
	const WCHAR* GenHitMissShaderEntryPoint[] = { MissShaderEntryPoint.c_str(), ClosestHitShaderEntryPoint.c_str(), RayGenShaderEntryPoint.c_str(), PlaneCHS.c_str() };
	D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION GenHitMissAssociation = {};
	GenHitMissAssociation.NumExports = _countof(GenHitMissShaderEntryPoint);
	GenHitMissAssociation.pExports = GenHitMissShaderEntryPoint;
	GenHitMissAssociation.pSubobjectToAssociate = &(Subobjects[Index - 1]);

	D3D12_STATE_SUBOBJECT GenHitMissSubobject = { D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION, &GenHitMissAssociation };
	Subobjects[Index++] = GenHitMissSubobject;

	// 11 Create pipeline config
	D3D12_RAYTRACING_PIPELINE_CONFIG PipelineConfig = { 1 };

	D3D12_STATE_SUBOBJECT PipelineSubobject = { D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &PipelineConfig };
	Subobjects[Index++] = PipelineSubobject;

	// 12 Create the global root signature and store the empty signature
	ComPtr<ID3DBlob> pGlobalSignatureBlob, pGlobalErrorBlob;
	CD3DX12_ROOT_SIGNATURE_DESC GlobalRootSignatureDesc = {};
	debug::check(D3D12SerializeRootSignature(&GlobalRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &pGlobalSignatureBlob, &pGlobalErrorBlob));
	debug::check(pDevice->CreateRootSignature(0, pGlobalSignatureBlob->GetBufferPointer(), pGlobalSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&m_pGlobalRootSignature)));
	ID3D12RootSignature* pGlobalRootSignature = m_pGlobalRootSignature.Get();

	D3D12_STATE_SUBOBJECT GlobalRootSignatureSubobject = { D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &pGlobalRootSignature };
	Subobjects[Index++] = GlobalRootSignatureSubobject;

	// Create PSO
	D3D12_STATE_OBJECT_DESC Desc = { D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, NUM_ARRAY_ARGS(Subobjects) };
	debug::check(pDevice->CreateStateObject(&Desc, IID_PPV_ARGS(&m_pPipelineState)));
}

void CRayTracingTriangle::_loadModels()
{
	auto pDevice = CIgniter::get()->fetchDevice();
	auto pCommandQueue = CIgniter::get()->fetchCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto pCommandList = pCommandQueue->createCommandList();
	
	// Upload vertex buffer
	UINT64 TriangleVertexBufferSize = _countof(TriangleVertices) * sizeof(SVertex);
	m_VertexBuffer[0].initAsBuffer(pCommandList.Get(), TriangleVertices, TriangleVertexBufferSize, D3D12_RESOURCE_STATE_GENERIC_READ);
	m_VertexBufferView[0].BufferLocation = m_VertexBuffer[0].getGPUVirtualAddress();
	m_VertexBufferView[0].SizeInBytes = sizeof(TriangleVertices);
	m_VertexBufferView[0].StrideInBytes = sizeof(SVertex);
	
	UINT64 PlaneVertexBufferSize = _countof(PlaneVertices) * sizeof(SVertex);
	m_VertexBuffer[1].initAsBuffer(pCommandList.Get(), PlaneVertices, PlaneVertexBufferSize, D3D12_RESOURCE_STATE_GENERIC_READ);
	m_VertexBufferView[1].BufferLocation = m_VertexBuffer[1].getGPUVirtualAddress();
	m_VertexBufferView[1].SizeInBytes = sizeof(PlaneVertices);
	m_VertexBufferView[1].StrideInBytes = sizeof(SVertex);
	
	SAccelerationStructureBuffers TriangleBLASBuffers;
	{
		D3D12_RAYTRACING_GEOMETRY_DESC GeometryDesc = {};
		GeometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
		GeometryDesc.Triangles.VertexBuffer.StartAddress = m_VertexBuffer[0].getGPUVirtualAddress();
		GeometryDesc.Triangles.VertexBuffer.StrideInBytes = sizeof(XMFLOAT3);
		GeometryDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
		GeometryDesc.Triangles.VertexCount = _countof(TriangleVertices);
		GeometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS Inputs = {};
		Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
		Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
		Inputs.NumDescs = 1;
		Inputs.pGeometryDescs = &GeometryDesc;
		Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;

		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO Info = {};
		pDevice->GetRaytracingAccelerationStructurePrebuildInfo(&Inputs, &Info);

		debug::check(pDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(Info.ScratchDataSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&(TriangleBLASBuffers.pScratch))));
		debug::check(pDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(Info.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&(TriangleBLASBuffers.pResult))));

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC ASDesc = {};
		ASDesc.Inputs = Inputs;
		ASDesc.DestAccelerationStructureData = TriangleBLASBuffers.pResult->GetGPUVirtualAddress();
		ASDesc.ScratchAccelerationStructureData = TriangleBLASBuffers.pScratch->GetGPUVirtualAddress();

		pCommandList->BuildRaytracingAccelerationStructure(&ASDesc, 0, nullptr);
		
		pCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(TriangleBLASBuffers.pResult.Get()));
	}
	m_pTriangleBLAS = TriangleBLASBuffers.pResult;

	SAccelerationStructureBuffers PlaneBLASBuffers;
	{
		D3D12_RAYTRACING_GEOMETRY_DESC GeometryDesc[2];

		GeometryDesc[0] = {};
		GeometryDesc[0].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
		GeometryDesc[0].Triangles.VertexBuffer.StartAddress = m_VertexBuffer[0].getGPUVirtualAddress();
		GeometryDesc[0].Triangles.VertexBuffer.StrideInBytes = sizeof(XMFLOAT3);
		GeometryDesc[0].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
		GeometryDesc[0].Triangles.VertexCount = _countof(TriangleVertices);
		GeometryDesc[0].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

		GeometryDesc[1] = {};
		GeometryDesc[1].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
		GeometryDesc[1].Triangles.VertexBuffer.StartAddress = m_VertexBuffer[1].getGPUVirtualAddress();
		GeometryDesc[1].Triangles.VertexBuffer.StrideInBytes = sizeof(XMFLOAT3);
		GeometryDesc[1].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
		GeometryDesc[1].Triangles.VertexCount = _countof(PlaneVertices);
		GeometryDesc[1].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS Inputs = {};
		Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
		Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
		Inputs.NumDescs = 2;
		Inputs.pGeometryDescs = GeometryDesc;
		Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;

		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO Info = {};
		pDevice->GetRaytracingAccelerationStructurePrebuildInfo(&Inputs, &Info);

		debug::check(pDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(Info.ScratchDataSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&(PlaneBLASBuffers.pScratch))));
		debug::check(pDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(Info.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&(PlaneBLASBuffers.pResult))));

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC ASDesc = {};
		ASDesc.Inputs = Inputs;
		ASDesc.DestAccelerationStructureData = PlaneBLASBuffers.pResult->GetGPUVirtualAddress();
		ASDesc.ScratchAccelerationStructureData = PlaneBLASBuffers.pScratch->GetGPUVirtualAddress();

		pCommandList->BuildRaytracingAccelerationStructure(&ASDesc, 0, nullptr);

		pCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(PlaneBLASBuffers.pResult.Get()));
	}
	m_pPlaneBLAS = PlaneBLASBuffers.pResult;
	
	SAccelerationStructureBuffers TLASBuffers;
	{
		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS Inputs = {};
		Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
		Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
		Inputs.NumDescs = 3;
		Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;

		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO Info;
		pDevice->GetRaytracingAccelerationStructurePrebuildInfo(&Inputs, &Info);

		debug::check(pDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(Info.ScratchDataSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&(TLASBuffers.pScratch))));
		debug::check(pDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(Info.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&(TLASBuffers.pResult))));
		debug::check(pDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * 3, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&(TLASBuffers.pInstanceDesc))));
		TLASBuffers.pInstanceDesc->SetName(L"InstanceDesc");

		D3D12_RAYTRACING_INSTANCE_DESC* pInstanceDesc;
		TLASBuffers.pInstanceDesc->Map(0, nullptr, (void**)&pInstanceDesc);

		XMMATRIX Matrixes[3];
		Matrixes[0] = XMMatrixTranspose(XMMatrixIdentity());
		Matrixes[1] = XMMatrixTranspose(XMMatrixTranslation(-2, 0, 0));
		Matrixes[2] = XMMatrixTranspose(XMMatrixTranslation( 2, 0, 0));

		pInstanceDesc[0].InstanceID = 0;
		pInstanceDesc[0].InstanceContributionToHitGroupIndex = 0;
		pInstanceDesc[0].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
		memcpy(pInstanceDesc[0].Transform, &(Matrixes[0]), sizeof(pInstanceDesc[0].Transform));
		pInstanceDesc[0].AccelerationStructure = PlaneBLASBuffers.pResult->GetGPUVirtualAddress();
		pInstanceDesc[0].InstanceMask = 0xFF;
		
		for (int i = 1; i < 3; i++)
		{
			pInstanceDesc[i].InstanceID = i;
			pInstanceDesc[i].InstanceContributionToHitGroupIndex = i + 1;
			pInstanceDesc[i].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
			memcpy(pInstanceDesc[i].Transform, &(Matrixes[i]), sizeof(pInstanceDesc[i].Transform));
			pInstanceDesc[i].AccelerationStructure = TriangleBLASBuffers.pResult->GetGPUVirtualAddress();
			pInstanceDesc[i].InstanceMask = 0xFF;
		}

		TLASBuffers.pInstanceDesc->Unmap(0, nullptr);

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC TLASDesc = {};
		TLASDesc.Inputs = Inputs;
		TLASDesc.Inputs.InstanceDescs = TLASBuffers.pInstanceDesc->GetGPUVirtualAddress();
		TLASDesc.DestAccelerationStructureData = TLASBuffers.pResult->GetGPUVirtualAddress();
		TLASDesc.ScratchAccelerationStructureData = TLASBuffers.pScratch->GetGPUVirtualAddress();

		pCommandList->BuildRaytracingAccelerationStructure(&TLASDesc, 0, nullptr);

		pCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(TLASBuffers.pResult.Get()));
	}
	m_pTLAS = TLASBuffers.pResult;

	CD3DX12_RESOURCE_DESC OutputDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, config::window::Width, config::window::Height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	m_pOutput.init(OutputDesc, D3D12_RESOURCE_STATE_COPY_SOURCE);

	for (int i = 0; i < 3; i++) {
		debug::check(pDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(sizeof(ColorData)), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&(m_pColor[i]))));
		UINT8* pColorData;
		debug::check(m_pColor[i]->Map(0, nullptr, (void**)&pColorData));
		memcpy(pColorData, ColorData + 3 * i, sizeof(ColorData));
		m_pColor[i]->Unmap(0, nullptr);
	}
	
	auto FenceValue = pCommandQueue->executeCommandList(pCommandList);
	pCommandQueue->wait4Fence(FenceValue);
	pCommandQueue->Flush();
}

void CRayTracingTriangle::_describeAssets()
{
	createDescriptorHeap(1, 1, 1, 0, 0);
	auto pDevice = CIgniter::get()->fetchDevice();

	D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
	UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	createUnorderedAccessDescriptor(0, UAVDesc, m_pOutput.fetchResourcePtr(), nullptr);

	D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
	SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
	SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	SRVDesc.RaytracingAccelerationStructure.Location = m_pTLAS->GetGPUVirtualAddress();
	createShaderResourceDescriptor(1, SRVDesc, nullptr);

	ComPtr<ID3D12StateObjectProperties> pRTSOProps;
	m_pPipelineState->QueryInterface(IID_PPV_ARGS(&pRTSOProps));
	m_ShaderTable.addEntry(SShaderTableEntry(pRTSOProps->GetShaderIdentifier(RayGenShaderEntryPoint.c_str()), m_CBVSRVUAVDescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr));
	m_ShaderTable.addEntry(SShaderTableEntry(pRTSOProps->GetShaderIdentifier(MissShaderEntryPoint.c_str()), 0));
	m_ShaderTable.addEntry(SShaderTableEntry(pRTSOProps->GetShaderIdentifier(HitGroupEntryPoint.c_str()), m_pColor[0]->GetGPUVirtualAddress()));
	m_ShaderTable.addEntry(SShaderTableEntry(pRTSOProps->GetShaderIdentifier(PlaneHitGroupEntryPoint.c_str()), 0));
	m_ShaderTable.addEntry(SShaderTableEntry(pRTSOProps->GetShaderIdentifier(HitGroupEntryPoint.c_str()), m_pColor[1]->GetGPUVirtualAddress()));
	m_ShaderTable.addEntry(SShaderTableEntry(pRTSOProps->GetShaderIdentifier(HitGroupEntryPoint.c_str()), m_pColor[2]->GetGPUVirtualAddress()));
	m_ShaderTable.uploadShaderTable();
	m_ShaderTableEntrySize = sizeof(SShaderTableEntry);
}
