#version 330 core

layout(location = 0) in vec3 P;

uniform mat4 model;

void main() {
  gl_Position = model * vec4(P, 1.0f);
}