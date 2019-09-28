#pragma once
#include "Config.h"
#include "Device.h"

namespace LearnVulkan
{
	class Buffer
	{
	public:
		Buffer(vk::UniqueBuffer&& vBuffer, vk::UniqueDeviceMemory&& vMemory, const void* vData, uint32_t vSize, Device* vDevice) : m_Buffer(std::move(vBuffer)), m_Memory(std::move(vMemory)), m_pDevice(vDevice)
		{
			void* Data;
			vkMapMemory(m_pDevice->fetchDevice(), m_Memory.get(), 0, vSize, 0, &Data);
			memcpy(Data, vData, vSize);
			vkUnmapMemory(m_pDevice->fetchDevice(), m_Memory.get());
		}

		vk::Buffer fetchBuffer() { return m_Buffer.get(); }
		
	private:
		vk::UniqueBuffer m_Buffer;
		vk::UniqueDeviceMemory m_Memory;
		Device* m_pDevice = nullptr;

		friend class Device;
	};
}