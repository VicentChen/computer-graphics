# Learn WebGPU

## 环境配置
 - 以下均为在`learn-webgpu/`目录下执行
 - `git clone https://github.com/glfw/glfw.git`
 - 下载[webgpu](https://github.com/eliemichel/WebGPU-distribution/archive/refs/heads/wgpu.zip), 解压内容到`webgpu/`
 - 下载[glfw3webgpu](https://github.com/eliemichel/glfw3webgpu/releases/download/v1.0.1/glfw3webgpu-v1.0.1.zip), 解压内容到`glfw3webgpu`

---

# WebGPU
- WebGPU是一种统一了Metal, DirectX, Vulkan, OpenGL, 甚至是WebGL的图形API.
- 理论上而言, WebGPU更像是一个封装好的RHI层.

## 资源管理
- 每个WebGPU对象都内置了一个引用计数器, 所以WebGPU对象一般的生命周期是这样的
```cpp
WGPUSomething sth = wgpuCreateSomething(/* descriptor */);

// This means "increase the ref counter of the object sth by 1"
wgpuSomethingReference(sth);
// Now the reference is 2 (it is set to 1 at creation)

// This means "decrease the ref counter of the object sth by 1
// and if it gets down to 0 then destroy the object"
wgpuSomethingRelease(sth);
// Now the reference is back to 1, the object can still be used

// Release again
wgpuSomethingRelease(sth);
// Now the reference is down to 0, the object is destroyed and
// should no longer be used!
```