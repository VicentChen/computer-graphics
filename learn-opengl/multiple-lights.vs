#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texture_coordinate;

out vec3 norm;
out vec3 fragment_pos;
out vec2 tex_coord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
  gl_Position = projection * view * model * vec4(pos, 1.0f);
  norm = normalize(mat3(transpose(inverse(model))) * normal);
  fragment_pos = vec3(model * vec4(pos, 1.0f));
  tex_coord = texture_coordinate;
}