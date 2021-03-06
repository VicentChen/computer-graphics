#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D Texture;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0f);// * texture(Texture, fragTexCoord).rgb, 1.0);// vec4(fragTexCoord, 0.0, 1.0);//vec4(fragColor, 1.0);
}