#version 330 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;

out vec3 fragment_pos;
out vec3 norm;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
  gl_Position = projection * view * model * vec4(pos, 1.0f);
  fragment_pos = vec3(model * vec4(pos, 1.0f));
  norm = normalize(mat3(transpose(inverse(model))) * normal);
}