#include "WebGPU.h"
#include <glfw3webgpu.h>
#include <iostream>

namespace webgpu
{
    // ----- WGPUInstance wrapper ----- //

    Instance::Instance(const InstanceDescriptor& descriptor)
    {
        std::cout << "Instance Constructor" << std::endl;


        instance = wgpuCreateInstance(&descriptor);
        if (!instance) {
            std::cout << "Failed to create WGPU instance" << std::endl;
            return;
        }
    }

    Instance::~Instance()
    {
        wgpuInstanceRelease(instance);
    }

    // ----- WGPUSurface wrapper ----- //

    Surface::Surface(const Instance& instance, GLFWwindow* window)
    {
        std::cout << "Surface Constructor" << std::endl;

        surface = glfwGetWGPUSurface(instance.get(), window);
    }

    Surface::Surface(wgpu::Instance instance, GLFWwindow* window)
    {
        surface = glfwGetWGPUSurface(instance, window);
    }

    Surface::~Surface()
    {
        wgpuSurfaceRelease(surface);
    }

    // ----- WGPUAdapter wrapper ----- //

    Adapter::Adapter(const Instance& instance, const AdapterOptions& options) {
        std::cout << "Adapter Constructor" << std::endl;

        struct UserData {
            WGPUAdapter adapter = nullptr;
            bool requestEnd = false;
        };
        UserData userData;

        static auto OnAdapterRequestEnded = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, char const* message, void* pUserData) {
            UserData& userData = *reinterpret_cast<UserData*>(pUserData);
            if (status == WGPURequestAdapterStatus_Success) {
                userData.adapter = adapter;
            } else {
                std::cout << "Failed to request adapter: " << message << std::endl;
            }
            userData.requestEnd = true;
        };

        WGPUInstance wgpuInstance = instance.get();
        wgpuInstanceRequestAdapter(wgpuInstance, &options, OnAdapterRequestEnded, &userData);

        adapter = userData.adapter;
    }

    Adapter::Adapter(wgpu::Instance instance, const AdapterOptions& options) {

        struct UserData {
            WGPUAdapter adapter = nullptr;
            bool requestEnd = false;
        };
        UserData userData;

        static auto OnAdapterRequestEnded = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, char const* message, void* pUserData) {
            UserData& userData = *reinterpret_cast<UserData*>(pUserData);
            if (status == WGPURequestAdapterStatus_Success) {
                userData.adapter = adapter;
            } else {
                std::cout << "Failed to request adapter: " << message << std::endl;
            }
            userData.requestEnd = true;
        };

        wgpuInstanceRequestAdapter(instance, &options, OnAdapterRequestEnded, &userData);

        adapter = userData.adapter;
    }

    Adapter::~Adapter()
    {
        wgpuAdapterRelease(adapter);
    }

    SupportedLimits Adapter::getSupportedLimits() const {
        SupportedLimits supportedLimits;
        wgpuAdapterGetLimits(adapter, &supportedLimits);
        return supportedLimits;
    }

    // ----- WGPUFeatureName wrapper ----- //

    std::string getFeatureName(FeatureName name) {
        switch (name) {
            case 0x00000000: return "Undefined";
            case 0x00000001: return "DepthClipControl";
            case 0x00000002: return "Depth32FloatStencil8";
            case 0x00000003: return "TimestampQuery";
            case 0x00000004: return "PipelineStatisticsQuery";
            case 0x00000005: return "TextureCompressionBC";
            case 0x00000006: return "TextureCompressionETC2";
            case 0x00000007: return "TextureCompressionASTC";
            case 0x00000008: return "IndirectFirstInstance";
            case 0x00000009: return "ShaderF16";
            case 0x0000000A: return "RG11B10UfloatRenderable";
            case 0x0000000B: return "BGRA8UnormStorage";
            case 0x0000000C: return "Float32Filterable";
            default: return "Unknown";
        }
    }

    // ----- WGPUErrorType wrapper ----- //

    std::string getErrorTypeName(ErrorType type) {
        switch (type) {
            case 0x00000000: return "NoError";
            case 0x00000001: return "Validation";
            case 0x00000002: return "OutOfMemory";
            case 0x00000003: return "Internal";
            case 0x00000004: return "Unknown";
            case 0x00000005: return "DeviceLost";
            default: return "Unknown";
        }
    }

    // ----- WGPUDevice wrapper ----- //

    Device::Device(const Adapter& adapter, const DeviceDescriptor& descriptor) {
        std::cout << "Device Constructor" << std::endl;

        struct UserData {
            WGPUDevice device = nullptr;
            bool requestEnd = false;
        };

        static auto OnDeviceRequestEnded = [](WGPURequestDeviceStatus status, WGPUDevice device, char const* message, void* pUserData) {
            UserData& userData = *reinterpret_cast<UserData*>(pUserData);
            if (status == WGPURequestDeviceStatus_Success) {
                userData.device = device;
            } else {
                std::cout << "Failed to request device: " << message << std::endl;
            }
            userData.requestEnd = true;
        };

        UserData userData;
        wgpuAdapterRequestDevice(adapter.get(), &descriptor, OnDeviceRequestEnded, (void*)&userData);

        device = userData.device;

        static auto OnDeviceError = [](WGPUErrorType type, char const* message, void* pUserData) {
            std::cout << "Device error, type: " << type << "(" << getErrorTypeName(type) << ")";
            if (message) std::cout << ". message: " << message;
            std::cout << std::endl;
        };
        wgpuDeviceSetUncapturedErrorCallback(device, OnDeviceError, nullptr);
    }

    Device::Device(const wgpu::Adapter& adapter, const wgpu::DeviceDescriptor& descriptor) {
        struct UserData {
            WGPUDevice device = nullptr;
            bool requestEnd = false;
        };

        static auto OnDeviceRequestEnded = [](WGPURequestDeviceStatus status, WGPUDevice device, char const* message, void* pUserData) {
            UserData& userData = *reinterpret_cast<UserData*>(pUserData);
            if (status == WGPURequestDeviceStatus_Success) {
                userData.device = device;
            } else {
                std::cout << "Failed to request device: " << message << std::endl;
            }
            userData.requestEnd = true;
        };

        UserData userData;
        wgpuAdapterRequestDevice(adapter, &descriptor, OnDeviceRequestEnded, &userData);

        device = userData.device;

        static auto OnDeviceError = [](WGPUErrorType type, char const* message, void* pUserData) {
            std::cout << "Device error, type: " << type << "(" << getErrorTypeName(type) << ")";
            if (message) std::cout << ". message: " << message;
            std::cout << std::endl;
        };
        wgpuDeviceSetUncapturedErrorCallback(device, OnDeviceError, nullptr);
    }

    Device::~Device() {
        wgpuDeviceRelease(device);
    }

    // ----- WGPUCommandBuffer wrapper ----- //

    CommandBuffer::CommandBuffer(const CommandEncoder& commandEncoder, const CommandBufferDescriptor& descriptor) {
        commandBuffer = wgpuCommandEncoderFinish(commandEncoder.get(), &descriptor);
    }

    CommandBuffer::~CommandBuffer() {
        // Uncomment if we are using Dawn WebGPU
        // wgpuCommandBufferRelease(commandBuffer);
    }

    // ----- WGPURenderPassEncoder wrapper ----- //

    RenderPassEncoder::RenderPassEncoder(const CommandEncoder& commandEncoder, const RenderPassDescriptor& descriptor) {
        renderPassEncoder = wgpuCommandEncoderBeginRenderPass(commandEncoder.get(), &descriptor);
    }

    RenderPassEncoder::~RenderPassEncoder() {
        wgpuRenderPassEncoderRelease(renderPassEncoder);
    }

    void RenderPassEncoder::setPipeline(const RenderPipeline& pipeline) {
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipeline.get());
    }

    void RenderPassEncoder::setVertexBuffer(uint32_t slot, const Buffer &buffer, uint64_t offset, uint64_t size) {
        wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder, slot, buffer.get(), offset, size);
    }

    void RenderPassEncoder::draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
        wgpuRenderPassEncoderDraw(renderPassEncoder, vertexCount, instanceCount, firstVertex, firstInstance);
    }

    void RenderPassEncoder::endPass() {
        wgpuRenderPassEncoderEnd(renderPassEncoder);
    }

    // ----- WGPUCommandEncoder wrapper ----- //

    CommandEncoder::CommandEncoder(const Device& device, const CommandEncoderDescriptor& descriptor) {
        commandEncoder = wgpuDeviceCreateCommandEncoder(device.get(), &descriptor);
    }

    CommandEncoder::~CommandEncoder() {
        // Uncomment if we are using Dawn WebGPU
        // wgpuCommandEncoderRelease(commandEncoder);
    }

    void CommandEncoder::insertDebugMarker(const std::string& markerLabel) {
        wgpuCommandEncoderInsertDebugMarker(commandEncoder, markerLabel.c_str());
    }

    RenderPassEncoder CommandEncoder::beginRenderPass(const RenderPassDescriptor& descriptor) {
        return RenderPassEncoder(*this, descriptor);
    }

    CommandBuffer CommandEncoder::encode(const CommandBufferDescriptor& descriptor) {
        return CommandBuffer(*this, descriptor);
    }

    // ----- WGPUQueue wrapper ----- //

    Queue::Queue(const Device& device) {
        queue = wgpuDeviceGetQueue(device.get());

        static auto OnQueueWorkDone = [](WGPUQueueWorkDoneStatus status, void* pUserData) {
            std::cout << "Queue work done, status: " << status << std::endl;
        };
        wgpuQueueOnSubmittedWorkDone(queue, OnQueueWorkDone, nullptr);
    }

    Queue::~Queue() {
        wgpuQueueRelease(queue);
    }

    void Queue::writeBuffer(Buffer& buffer, int bufferOffset, const void* data, int size) {
        wgpuQueueWriteBuffer(queue, buffer.get(), bufferOffset, data, size);
    }

    void Queue::submit(std::vector<CommandBuffer> commandBuffers) {
        std::vector<WGPUCommandBuffer> buffers;
        buffers.reserve(commandBuffers.size());
        for (CommandBuffer& commandBuffer : commandBuffers) {
            buffers.push_back(commandBuffer.get());
        }
        wgpuQueueSubmit(queue, static_cast<uint32_t>(buffers.size()), buffers.data());
        std::cout << "Queue submit " << buffers.size() << " command buffers" << std::endl;
    }

    // ----- WGPUTextureView wrapper ----- //

    TextureView::TextureView(SwapChain& swapChain) {
        textureView = wgpuSwapChainGetCurrentTextureView(swapChain.get());
        if (!textureView) {
            std::cout << "Failed to get next texture view" << std::endl;
            return;
        }
    }

    TextureView::~TextureView() {
        wgpuTextureViewRelease(textureView);
    }

    // ----- WGPUSwapChain wrapper ----- //

    SwapChain::SwapChain(const Adapter& adapter, const Device& device, const Surface& surface, SwapChainDescriptor& descriptor) {
        format = wgpuSurfaceGetPreferredFormat(surface.get(), adapter.get());
        descriptor.format = format;
        descriptor.usage = WGPUTextureUsage_RenderAttachment;
        descriptor.presentMode = WGPUPresentMode_Fifo;

        swapChain = wgpuDeviceCreateSwapChain(device.get(), surface.get(), &descriptor);
    }

    SwapChain::SwapChain(wgpu::Adapter adapter, const Device& device, wgpu::Surface surface, SwapChainDescriptor& descriptor) {
        format = wgpuSurfaceGetPreferredFormat(surface, adapter);
        descriptor.format = format;
        descriptor.usage = WGPUTextureUsage_RenderAttachment;
        descriptor.presentMode = WGPUPresentMode_Fifo;

        swapChain = wgpuDeviceCreateSwapChain(device.get(), surface, &descriptor);
    }

    SwapChain::~SwapChain() {
        wgpuSwapChainRelease(swapChain);
    }

    TextureView SwapChain::getCurrentTextureView() {
        return TextureView(*this);
    }

    void SwapChain::present() {
        wgpuSwapChainPresent(swapChain);
    }

    // ----- WGPUShaderModule wrapper ----- //

    ShaderModule::ShaderModule(const Device& device, const ShaderModuleDescriptor& descriptor) {
        shaderModule = wgpuDeviceCreateShaderModule(device.get(), &descriptor);
    }

    ShaderModule::~ShaderModule() {
        wgpuShaderModuleRelease(shaderModule);
    }

    // ----- WGPURenderPipeline wrapper ----- //

    RenderPipeline::RenderPipeline(const Device& device, const RenderPipelineDescriptor& descriptor) {
        renderPipeline = wgpuDeviceCreateRenderPipeline(device.get(), &descriptor);
    }

    RenderPipeline::~RenderPipeline() {
        wgpuRenderPipelineRelease(renderPipeline);
    }

    // ----- WGPUBuffer wrapper ----- //

    Buffer::Buffer(const Device& device, const BufferDescriptor& descriptor) {
        buffer = wgpuDeviceCreateBuffer(device.get(), &descriptor);
    }

    Buffer::~Buffer() {
        wgpuBufferRelease(buffer);
    }
} // namespace webgpu