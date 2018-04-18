#version 330 core

layout(location = 0) in vec3 P;
layout(location = 1) in vec3 N;
layout(location = 2) in vec2 T;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform bool reverse_normals;

out vec3 normal;
out vec2 tex_coord;
out vec3 fragment_pos;

void main() {
  gl_Position = projection * view * model * vec4(P, 1.0f);
  if (reverse_normals) {
    normal = normalize(transpose(inverse(mat3(model))) * (-1.0f * N));
  }
  else {
    normal = normalize(transpose(inverse(mat3(model))) * N);
  }
  tex_coord = T;
  fragment_pos = vec3(model * vec4(P, 1.0f));
}
