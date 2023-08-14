#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>
#include "WebGPU.h"
#include <iostream>
#include <vector>


int main(int argc, char** argv) {

    const int ScreenWidth = 1920;
    const int ScreenHeight = 1080;

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(ScreenWidth, ScreenHeight, "LearnWebGPU", NULL, NULL);

//    webgpu::InstanceDescriptor descriptor = {};
//    descriptor.nextInChain = nullptr;
//    webgpu::Instance instance(descriptor);
//    std::cout << "GPU instance: " << instance.get() << std::endl;

    webgpu::InstanceDescriptor descriptor = {};
//    WGPUInstance instance = wgpuCreateInstance(&descriptor);
    webgpu::Instance instance(descriptor);

    webgpu::Surface surface(instance, window);
    std::cout << "GPU surface: " << surface.get() << std::endl;
//    wgpu::Surface surface = glfwGetWGPUSurface(instance, window);


    webgpu::AdapterOptions adapterOptions = {};
    adapterOptions.nextInChain = nullptr;
    adapterOptions.compatibleSurface = surface.get();
    WGPUInstance wgpuInstance = instance.get(); // BUG: I really cannot understand why removing this line will cause a random memory access crash.
    webgpu::Adapter adapter(instance, adapterOptions);
    std::cout << "GPU adapter: " << adapter.get() << std::endl;
    webgpu::SupportedLimits supportedLimits = adapter.getSupportedLimits();

    std::vector<webgpu::FeatureName> features;
    int featureNum = static_cast<int>(wgpuAdapterEnumerateFeatures(adapter.get(), nullptr));
    features.resize(featureNum);
    wgpuAdapterEnumerateFeatures(adapter.get(), features.data());
    for (webgpu::FeatureName feature : features) {
        std::cout << "GPU feature: " << feature << "--" << webgpu::getFeatureName(feature) << std::endl;
    }

    webgpu::RequiredLimits requiredLimits = {};
    requiredLimits.limits.maxTextureDimension1D = 0;
    requiredLimits.limits.maxTextureDimension2D = 0;
    requiredLimits.limits.maxTextureDimension3D = 0;
    requiredLimits.limits.maxTextureArrayLayers = 0;
    requiredLimits.limits.maxBindGroups = 0;
    requiredLimits.limits.maxBindingsPerBindGroup = 0;
    requiredLimits.limits.maxDynamicUniformBuffersPerPipelineLayout = 0;
    requiredLimits.limits.maxDynamicStorageBuffersPerPipelineLayout = 0;
    requiredLimits.limits.maxSampledTexturesPerShaderStage = 0;
    requiredLimits.limits.maxSamplersPerShaderStage = 0;
    requiredLimits.limits.maxStorageBuffersPerShaderStage = 0;
    requiredLimits.limits.maxStorageTexturesPerShaderStage = 0;
    requiredLimits.limits.maxUniformBuffersPerShaderStage = 0;
    requiredLimits.limits.maxUniformBufferBindingSize = 0;
    requiredLimits.limits.maxStorageBufferBindingSize = 0;
    requiredLimits.limits.minUniformBufferOffsetAlignment = 64;
    requiredLimits.limits.minStorageBufferOffsetAlignment = 16;
    requiredLimits.limits.maxVertexBuffers = 0;
    requiredLimits.limits.maxBufferSize = 0;
    requiredLimits.limits.maxVertexAttributes = 0;
    requiredLimits.limits.maxVertexBufferArrayStride = 0;
    requiredLimits.limits.maxInterStageShaderComponents = 0;
    requiredLimits.limits.maxInterStageShaderVariables = 0;
    requiredLimits.limits.maxColorAttachments = 0;
    requiredLimits.limits.maxColorAttachmentBytesPerSample = 0;
    requiredLimits.limits.maxComputeWorkgroupStorageSize = 0;
    requiredLimits.limits.maxComputeInvocationsPerWorkgroup = 0;
    requiredLimits.limits.maxComputeWorkgroupSizeX = 0;
    requiredLimits.limits.maxComputeWorkgroupSizeY = 0;
    requiredLimits.limits.maxComputeWorkgroupSizeZ = 0;
    requiredLimits.limits.maxComputeWorkgroupsPerDimension = 0;

    requiredLimits.limits.maxVertexAttributes = 1;
    requiredLimits.limits.maxVertexBuffers = 1;
    requiredLimits.limits.maxBufferSize = 6 * 2 * sizeof(float);
    requiredLimits.limits.maxVertexBufferArrayStride = 2 * sizeof(float);
    requiredLimits.limits.minUniformBufferOffsetAlignment = supportedLimits.limits.minUniformBufferOffsetAlignment;
    requiredLimits.limits.minStorageBufferOffsetAlignment = supportedLimits.limits.minStorageBufferOffsetAlignment;

    std::string deviceLabel = "LearnWebGPU";
    std::string defaultQueueLabel = "LearnWebGPU Defaylt Queue";
    webgpu::DeviceDescriptor deviceDescriptor = {};
    deviceDescriptor.label = deviceLabel.c_str();
    deviceDescriptor.requiredFeaturesCount = 0;
    deviceDescriptor.requiredLimits = &requiredLimits;
    deviceDescriptor.defaultQueue.label = defaultQueueLabel.c_str();
    deviceDescriptor.defaultQueue.nextInChain = nullptr;
    webgpu::Device device(adapter, deviceDescriptor);
    std::cout << "GPU device: " << device.get() << std::endl;

    webgpu::Queue queue(device);
    std::cout << "GPU queue: " << queue.get() << std::endl;

    webgpu::SwapChainDescriptor swapChainDescriptor = {};
    swapChainDescriptor.nextInChain = nullptr;
    swapChainDescriptor.label = "LearnWebGPU Swap Chain";
    swapChainDescriptor.width = ScreenWidth;
    swapChainDescriptor.height = ScreenHeight;
    webgpu::SwapChain swapChain(adapter, device, surface, swapChainDescriptor);

    webgpu::VertexAttribute vertexAttribute = {};
    vertexAttribute.shaderLocation = 0;
    vertexAttribute.format = WGPUVertexFormat_Float32x2;
    vertexAttribute.offset = 0;

    constexpr int vertexAttributeFloatNum = 2;

    webgpu::VertexBufferlayout vertexBufferLayout = {};
    vertexBufferLayout.attributeCount = 1;
    vertexBufferLayout.attributes = &vertexAttribute;
    vertexBufferLayout.arrayStride = vertexAttributeFloatNum * sizeof(float);
    vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;

    std::vector<float> vertexData = {
            -0.5, -0.5,
            +0.5, -0.5,
            +0.0, +0.5,

            -0.55f, -0.5,
            -0.05f, +0.5,
            -0.55f, +0.5
    };
    webgpu::BufferDescriptor vertexBufferDescriptor = {};
    vertexBufferDescriptor.nextInChain = nullptr;
    vertexBufferDescriptor.label = "LearnWebGPU Vertex Buffer";
    vertexBufferDescriptor.size = vertexData.size() * sizeof(float);
    vertexBufferDescriptor.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    vertexBufferDescriptor.mappedAtCreation = false;
    webgpu::Buffer vertexBuffer(device, vertexBufferDescriptor);
    queue.writeBuffer(vertexBuffer, 0, vertexData.data(), vertexBufferDescriptor.size);

    std::string shaderSource = ""
                               "@vertex\n"
                               "fn vs_main(@location(0) in_vertex_position : vec2f) -> @builtin(position) vec4f {\n"
                               "  return vec4f(in_vertex_position, 0.0f, 1.0f);\n"
                               "}\n"
                               "\n"
                               "@fragment\n"
                               "fn fs_main() -> @location(0) vec4f {\n"
                               "  return vec4f(0.8f, 0.7f, 0.6f, 1.0f);\n"
                               "}\n"
                               "";

    webgpu::ShaderModuleWGSLDescriptor shaderModuleWGSLDescriptor = {};
    shaderModuleWGSLDescriptor.chain.next = nullptr;
    shaderModuleWGSLDescriptor.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    shaderModuleWGSLDescriptor.code = shaderSource.c_str();

    webgpu::ShaderModuleDescriptor shaderModuleDescriptor = {};
    shaderModuleDescriptor.hintCount = 0;
    shaderModuleDescriptor.hints = nullptr;
    shaderModuleDescriptor.nextInChain = &shaderModuleWGSLDescriptor.chain;
    webgpu::ShaderModule shaderModule(device, shaderModuleDescriptor);

    webgpu::RenderPipelineDescriptor renderPipelineDescriptor = {};
    renderPipelineDescriptor.nextInChain = nullptr;
    renderPipelineDescriptor.vertex.bufferCount = 1;
    renderPipelineDescriptor.vertex.buffers = &vertexBufferLayout;
    renderPipelineDescriptor.vertex.module = shaderModule.get();
    renderPipelineDescriptor.vertex.entryPoint = "vs_main";
    renderPipelineDescriptor.vertex.constantCount = 0;
    renderPipelineDescriptor.vertex.constants = nullptr;

    renderPipelineDescriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    renderPipelineDescriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    renderPipelineDescriptor.primitive.frontFace = WGPUFrontFace_CCW;
    renderPipelineDescriptor.primitive.cullMode = WGPUCullMode_None;

    webgpu::BlendState blendState = {};
    blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blendState.color.operation = WGPUBlendOperation_Add;
    blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
    blendState.alpha.dstFactor = WGPUBlendFactor_One;
    blendState.alpha.operation = WGPUBlendOperation_Add;

    webgpu::ColorTargetState colorTargetState = {};
    colorTargetState.format = swapChain.format;
    colorTargetState.blend = &blendState;
    colorTargetState.writeMask = WGPUColorWriteMask_All;

    webgpu::FragmentState fragmentState = {};
    fragmentState.module = shaderModule.get();
    fragmentState.entryPoint = "fs_main";
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTargetState;
    renderPipelineDescriptor.fragment = &fragmentState;

    renderPipelineDescriptor.depthStencil = nullptr;
    renderPipelineDescriptor.multisample.count = 1;
    renderPipelineDescriptor.multisample.mask = ~0u;
    renderPipelineDescriptor.multisample.alphaToCoverageEnabled = false;

    renderPipelineDescriptor.layout = nullptr;

    webgpu::RenderPipeline renderPipeline(device, renderPipelineDescriptor);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        std::cout << "Frame" << std::endl;

        webgpu::TextureView currentTextureView = swapChain.getCurrentTextureView();

        webgpu::CommandEncoderDescriptor commandEncoderDescriptor = {};
        commandEncoderDescriptor.nextInChain = nullptr;
        commandEncoderDescriptor.label = "LearnWebGPU Command Encoder";
        webgpu::CommandEncoder commandEncoder(device, commandEncoderDescriptor);

        std::vector<webgpu::RenderPassColorAttachment> colorAttachment = { {} };
        colorAttachment[0].view = currentTextureView.get();
        colorAttachment[0].resolveTarget = nullptr;
        colorAttachment[0].loadOp = WGPULoadOp_Clear;
        colorAttachment[0].storeOp = WGPUStoreOp_Store;
        colorAttachment[0].clearValue = { 0.5f, 0.7f, 0.7f, 1.0f };

        webgpu::RenderPassDescriptor renderPassDescriptor = {};
        renderPassDescriptor.colorAttachmentCount = static_cast<uint32_t>(colorAttachment.size());
        renderPassDescriptor.colorAttachments = colorAttachment.data();
        renderPassDescriptor.depthStencilAttachment = nullptr;
        renderPassDescriptor.timestampWriteCount = 0;
        renderPassDescriptor.timestampWrites = nullptr;
        renderPassDescriptor.nextInChain = nullptr;
        webgpu::RenderPassEncoder renderPass = commandEncoder.beginRenderPass(renderPassDescriptor);
        renderPass.setPipeline(renderPipeline);
        renderPass.setVertexBuffer(0, vertexBuffer, 0, vertexData.size() * sizeof(float));
        renderPass.draw(vertexData.size() / vertexAttributeFloatNum, 1, 0, 0);
        renderPass.endPass();

        webgpu::CommandBufferDescriptor commandBufferDescriptor = {};
        commandBufferDescriptor.nextInChain = nullptr;
        commandBufferDescriptor.label = "LearnWebGPU Command Buffer";
        webgpu::CommandBuffer commandBuffer(commandEncoder, commandBufferDescriptor);

        std::vector<webgpu::CommandBuffer> commandBuffers = {commandBuffer};
        queue.submit(commandBuffers);

        swapChain.present();
    }

    glfwDestroyWindow(window);

    return 0;
}
