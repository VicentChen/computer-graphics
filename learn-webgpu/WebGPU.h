#include <webgpu/webgpu.h>
#include <GLFW/glfw3.h>
#include <string>

#include <webgpu/webgpu.hpp>
#include <glfw3webgpu.h>

namespace webgpu
{
    struct CommandEncoder;
    struct RenderPipeline;
    struct SwapChain;
    struct Buffer;

    using InstanceDescriptor = WGPUInstanceDescriptor;
    struct Instance {
        WGPUInstance instance = nullptr;

        explicit Instance(const InstanceDescriptor& descriptor);
        ~Instance();

        WGPUInstance operator->() const { return instance; }
        WGPUInstance get() const { return instance; }
    };

    struct Surface {
        WGPUSurface surface = nullptr;

        explicit Surface(const Instance& instance, GLFWwindow* window);
        explicit Surface(wgpu::Instance instance, GLFWwindow* window);
        ~Surface();

        WGPUSurface operator->() const { return surface; }
        WGPUSurface get() const { return surface; }
    };

    using Limits = WGPULimits;
    using SupportedLimits = WGPUSupportedLimits;
    using RequiredLimits = WGPURequiredLimits;
    using AdapterOptions = WGPURequestAdapterOptions;
    struct Adapter {
        WGPUAdapter adapter = nullptr;

        explicit Adapter(const Instance& instance, const AdapterOptions& options);
        explicit Adapter(wgpu::Instance instance, const AdapterOptions& options);
        ~Adapter();

        SupportedLimits getSupportedLimits() const;

        WGPUAdapter operator->() const { return adapter; }
        WGPUAdapter get() const { return adapter; }
    };

    using FeatureName = WGPUFeatureName;
    std::string getFeatureName(FeatureName featureName);

    using ErrorType = WGPUErrorType;
    std::string getErrorTypeName(ErrorType errorType);

    using DeviceDescriptor = WGPUDeviceDescriptor;
    struct Device {
        WGPUDevice device = nullptr;

        explicit Device(const Adapter& adapter, const DeviceDescriptor& descriptor);
        explicit Device(const wgpu::Adapter& adapter, const wgpu::DeviceDescriptor& descriptor);
        ~Device();

        WGPUDevice operator->() const { return device; }
        WGPUDevice get() const { return device; }
    };


    using CommandBufferDescriptor = WGPUCommandBufferDescriptor;
    struct CommandBuffer {
        WGPUCommandBuffer commandBuffer = nullptr;

        CommandBuffer(const CommandEncoder& commandEncoder, const CommandBufferDescriptor& descriptor);
        ~CommandBuffer();

        WGPUCommandBuffer operator->() const { return commandBuffer; }
        WGPUCommandBuffer get() const { return commandBuffer; }
    };


    using RenderPassColorAttachment = WGPURenderPassColorAttachment;
    using RenderPassDescriptor = WGPURenderPassDescriptor;
    struct RenderPassEncoder {
        WGPURenderPassEncoder renderPassEncoder = nullptr;

        RenderPassEncoder(const CommandEncoder& commandEncoder, const RenderPassDescriptor& descriptor);
        ~RenderPassEncoder();

        void setPipeline(const RenderPipeline& pipeline);
        void setVertexBuffer(uint32_t slot, const Buffer& buffer, uint64_t offset, uint64_t size);
        void draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance);
        void endPass();

        WGPURenderPassEncoder operator->() const { return renderPassEncoder; }
        WGPURenderPassEncoder get() const { return renderPassEncoder; }
    };

    using CommandEncoderDescriptor = WGPUCommandEncoderDescriptor;
    struct CommandEncoder {
        WGPUCommandEncoder commandEncoder = nullptr;

        CommandEncoder(const Device& device, const CommandEncoderDescriptor& descriptor);
        ~CommandEncoder();

        void insertDebugMarker(const std::string& markerLabel);
        CommandBuffer encode(const CommandBufferDescriptor& descriptor);
        RenderPassEncoder beginRenderPass(const RenderPassDescriptor& descriptor);

        WGPUCommandEncoder operator->() const { return commandEncoder; }
        WGPUCommandEncoder get() const { return commandEncoder; }
    };

    struct Queue {
        WGPUQueue queue = nullptr;

        Queue(const Device& device);
        ~Queue();

        void writeBuffer(Buffer& buffer, int bufferOffset, const void* data, int size);
        void submit(std::vector<CommandBuffer> commandBuffers);

        WGPUQueue operator->() const { return queue; }
        WGPUQueue get() const { return queue; }
    };

    using TextureFormat = WGPUTextureFormat;
    struct TextureView {
        WGPUTextureView textureView = nullptr;

        TextureView(SwapChain& swapChain);
        ~TextureView();

        WGPUTextureView operator->() const { return textureView; }
        WGPUTextureView get() const { return textureView; }
    };

    using SwapChainDescriptor = WGPUSwapChainDescriptor;
    struct SwapChain {
        WGPUSwapChain swapChain = nullptr;
        TextureFormat format = WGPUTextureFormat_Undefined;

        SwapChain(const Adapter& adapter, const Device& device, const Surface& surface, SwapChainDescriptor& descriptor);
        SwapChain(wgpu::Adapter adapter, const Device& device, wgpu::Surface surface, SwapChainDescriptor& descriptor);
        ~SwapChain();

        TextureView getCurrentTextureView();
        void present();

        WGPUSwapChain operator->() const { return swapChain; }
        WGPUSwapChain get() const { return swapChain; }
    };

    using ShaderModuleDescriptor = WGPUShaderModuleDescriptor;
    using ShaderModuleWGSLDescriptor = WGPUShaderModuleWGSLDescriptor;
    struct ShaderModule {
        WGPUShaderModule shaderModule = nullptr;

        ShaderModule(const Device& device, const ShaderModuleDescriptor& descriptor);
        ~ShaderModule();

        WGPUShaderModule operator->() const { return shaderModule; }
        WGPUShaderModule get() const { return shaderModule; }
    };

    using FragmentState = WGPUFragmentState;
    using DepthStencilState = WGPUDepthStencilState;
    using BlendState = WGPUBlendState;
    using ColorTargetState = WGPUColorTargetState;
    using RenderPipelineDescriptor = WGPURenderPipelineDescriptor;
    struct RenderPipeline {
        WGPURenderPipeline renderPipeline = nullptr;

        RenderPipeline(const Device& device, const RenderPipelineDescriptor& descriptor);
        ~RenderPipeline();

        WGPURenderPipeline operator->() const { return renderPipeline; }
        WGPURenderPipeline get() const { return renderPipeline; }
    };

    using VertexAttribute = WGPUVertexAttribute;
    using VertexBufferlayout = WGPUVertexBufferLayout;
    using BufferDescriptor = WGPUBufferDescriptor;
    struct Buffer {
        WGPUBuffer buffer = nullptr;

        Buffer(const Device& device, const BufferDescriptor& descriptor);
        ~Buffer();

        WGPUBuffer operator->() const { return buffer; }
        WGPUBuffer get() const { return buffer; }
    };
} // namespace webgpu
