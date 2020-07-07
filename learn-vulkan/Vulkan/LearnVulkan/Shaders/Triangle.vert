#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform TransformMatrices {
	mat4 Model;
	mat4 View;
	mat4 Proj;
} tm;

layout(location = 0) in vec3 Vert;
layout(location = 1) in vec3 Color;
layout(location = 2) in vec2 TexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = tm.Proj * tm.View * tm.Model * vec4(Vert, 1.0);
    fragColor = Color;
	fragTexCoord = TexCoord;
}