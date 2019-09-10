#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject
{
	mat4 Model;
	mat4 View;
	mat4 Proj;
} Ubo;

layout(location = 0) in vec3 InPos;
layout(location = 1) in vec3 InColor;
layout(location = 2) in vec2 InTexCoord;

layout(location = 0) out vec3 FragColor;
layout(location = 1) out vec2 FragTexCoord;

void main()
{
	gl_Position = Ubo.Proj * Ubo.View * Ubo.Model * vec4(InPos, 1.0);
	FragColor = InColor;
	FragTexCoord = InTexCoord;
}