#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 Vert;
layout(location = 1) in vec3 Color;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = vec4(Vert, 1.0);
    fragColor = Color;
}