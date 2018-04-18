#define STB_IMAGE_IMPLEMENTATION

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <learnopengl/shader.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#define LIGHT_NUM 4
#define NEAR_PLANE 1.0f
#define FAR_PLANE 7.5f
#define SHADOW_WIDTH 1280
#define SHADOW_HEIGHT 720

// TODO: remove SHADOW_WIDTH, SHADOW_HEIGHT

using namespace glm;

void render_scene(Shader &shader, unsigned int texture, unsigned int depth_map, unsigned int plane_VAO, unsigned int cube_VAO);
void render_depth_map(Shader &depth_shader, unsigned int texture, unsigned int FBO, unsigned int plane_VAO, unsigned int cube_VAO);
void render_debug_scene(Shader &debug_shader, unsigned int depth_map, unsigned int quad_VAO);

void process_input(GLFWwindow *window);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void load_cube(unsigned int *VAO, unsigned int *VBO);
void load_plane(unsigned int *VAO, unsigned int *VBO);
void load_quad(unsigned int *VAO, unsigned int *VBO);
unsigned int load_texture(const char* path);
unsigned int load_depth_map();
unsigned int load_framebuffer(unsigned int depth_map);

// settings 
const int SCR_WIDTH = 1280;
const int SCR_HEIGHT = 720;

// camera
Camera camera(vec3(-2.0f, 4.0f, -1.0f));
float first_mouse = true;
float last_X = SCR_WIDTH / 2.0f;
float last_Y = SCR_HEIGHT / 2.0f;

// timing
float last_frame = 0.0f;
float curr_frame = 0.0f;
float delta_time = 0.0f;

// lighting
vec3 light_pos = vec3(-2.0f, 4.0f, -1.0f);
bool blinn = true;

int main(int argc, char *argv[]) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Shadow Mapping", NULL, NULL);
  glfwMakeContextCurrent(window);
  gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glEnable(GL_DEPTH_TEST); 

  // shaders
  Shader shader("shadow-mapping.vs", "shadow-mapping.fs");
  Shader depth_shader("depth-map.vs", "depth-map.fs");
  Shader debug_shader("debug.vs", "debug.fs");

  // vertices
  unsigned int plane_VAO, plane_VBO;
  load_plane(&plane_VAO, &plane_VBO);
  unsigned int cube_VAO, cube_VBO;
  load_cube(&cube_VAO, &cube_VBO);
  unsigned int quad_VAO, quad_VBO;
  load_quad(&quad_VAO, &quad_VBO);

  // textures
  unsigned int texture = load_texture("pic/wood.png");
  shader.use(); shader.setInt("texture", 0);
  unsigned int depth_map = load_depth_map();
  debug_shader.use(); debug_shader.setInt("depth_map", 0);
  shader.use(); shader.setInt("shadow_map", 1);

  // framebuffer
  unsigned int FBO = load_framebuffer(depth_map);

  while (!glfwWindowShouldClose(window)) {
    curr_frame = glfwGetTime();
    delta_time = curr_frame - last_frame;
    last_frame = curr_frame;

    process_input(window);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    render_depth_map(depth_shader, texture, FBO, plane_VAO, cube_VAO);
    //render_debug_scene(debug_shader, depth_map, quad_VAO);
    render_scene(shader, texture, depth_map, plane_VAO, cube_VAO);

    glfwPollEvents();
    glfwSwapBuffers(window);
  }

  glfwTerminate();
  return 0;
}

void render_depth_map(Shader &depth_shader, unsigned int texture, unsigned int FBO, unsigned int plane_VAO, unsigned int cube_VAO) {
  mat4 model, view, projection;
  model = mat4();
  view = lookAt(light_pos, vec3(0.0f), vec3(0.0f, 1.0f, 0.0f));
  projection = ortho(-10.0f, 10.0f, -10.0f, 10.0f, NEAR_PLANE, FAR_PLANE);

  depth_shader.use();

  // plane
  depth_shader.setMat4("model", model);
  depth_shader.setMat4("view", view);
  depth_shader.setMat4("projection", projection);

  glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
  glBindFramebuffer(GL_FRAMEBUFFER, FBO);
  glClear(GL_DEPTH_BUFFER_BIT);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glBindVertexArray(plane_VAO);
  glDrawArrays(GL_TRIANGLES, 0, 6);

  // cubes
  glBindVertexArray(cube_VAO);
  model = mat4(); model = translate(model, vec3(0.0f, 1.5f, 0.0f)); model = scale(model, vec3(0.5f));
  depth_shader.setMat4("model", model); glDrawArrays(GL_TRIANGLES, 0, 36);
  model = mat4(); model = translate(model, vec3(2.0f, 0.01f, 1.0)); model = scale(model, vec3(0.5f)); // avoid z-fighting
  depth_shader.setMat4("model", model); glDrawArrays(GL_TRIANGLES, 0, 36);
  model = mat4(); model = translate(model, vec3(-1.0f, 0.0f, 2.0)); model = rotate(model, radians(60.0f), normalize(vec3(1.0, 0.0, 1.0))); model = scale(model, vec3(0.25f));
  depth_shader.setMat4("model", model); glDrawArrays(GL_TRIANGLES, 0, 36);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glBindVertexArray(0);
  glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void render_scene(Shader &shader, unsigned int texture, unsigned int depth_map, unsigned int plane_VAO, unsigned int cube_VAO) {
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, depth_map);

  // light
  shader.use();
  shader.setBool("blinn", blinn);
  shader.setVec3("light_pos", light_pos);
  shader.setVec3("view_pos", camera.Position);

  mat4 model, view, projection, lightspace_view, lightspace_projection;
  model = mat4();
  view = camera.GetViewMatrix();
  projection = perspective(radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

  lightspace_view = lookAt(light_pos, vec3(0.0f), vec3(0.0f, 1.0f, 0.0f));
  lightspace_projection = ortho(-10.0f, 10.0f, -10.0f, 10.0f, NEAR_PLANE, FAR_PLANE);

  // plane
  shader.setMat4("model", model);
  shader.setMat4("view", view);
  shader.setMat4("projection", projection);
  shader.setMat4("lightspace_view", lightspace_view);
  shader.setMat4("lightspace_projection", lightspace_projection);
  glBindVertexArray(plane_VAO);
  glDrawArrays(GL_TRIANGLES, 0, 6);

  // cubes
  glBindVertexArray(cube_VAO);
  model = mat4(); model = translate(model, vec3(0.0f, 1.5f, 0.0f)); model = scale(model, vec3(0.5f));
  shader.setMat4("model", model); glDrawArrays(GL_TRIANGLES, 0, 36);
  model = mat4(); model = translate(model, vec3(2.0f, 0.0f, 1.0)); model = scale(model, vec3(0.5f)); // avoid z-fighting
  shader.setMat4("model", model); glDrawArrays(GL_TRIANGLES, 0, 36);
  model = mat4(); model = translate(model, vec3(-1.0f, 0.0f, 2.0)); model = rotate(model, radians(60.0f), normalize(vec3(1.0, 0.0, 1.0))); model = scale(model, vec3(0.25f));
  shader.setMat4("model", model); glDrawArrays(GL_TRIANGLES, 0, 36);
  glBindVertexArray(0);
}

void render_debug_scene(Shader &debug_shader, unsigned int depth_map, unsigned int quad_VAO) {
  debug_shader.use();
  debug_shader.setFloat("near_plane", NEAR_PLANE);
  debug_shader.setFloat("far_plane", FAR_PLANE);
  glBindVertexArray(quad_VAO);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, depth_map);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);
}

void process_input(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.ProcessKeyboard(FORWARD, delta_time);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.ProcessKeyboard(BACKWARD, delta_time);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.ProcessKeyboard(LEFT, delta_time);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.ProcessKeyboard(RIGHT, delta_time);
  blinn = (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) ? true : false;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
  if (first_mouse) {
    last_X = xpos;
    last_Y = ypos;
    first_mouse = false;
  }

  double xoffset = xpos - last_X;
  double yoffset = last_Y - ypos;
  last_X = xpos; last_Y = ypos;

  camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  camera.ProcessMouseScroll(yoffset);
}

void load_cube(unsigned int *VAO, unsigned int *VBO) {
  float vertices[] = {
    // back face
    -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
    1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
    1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
    1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
    -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
    -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
    // front face
    -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
    1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
    1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
    1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
    -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
    -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
    // left face
    -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
    -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
    -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
    -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
    -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
    -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
    // right face
    1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
    1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
    1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
    1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
    1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
    1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
    // bottom face
    -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
    1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
    1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
    1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
    -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
    -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
    // top face
    -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
    1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
    1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
    1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
    -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
    -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
  };
  glGenVertexArrays(1, VAO);
  glGenBuffers(1, VBO);
  // fill buffer
  glBindBuffer(GL_ARRAY_BUFFER, *VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  // link vertex attributes
  glBindVertexArray(*VAO);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void load_plane(unsigned int *VAO, unsigned int *VBO) {
  float vertices[] = {
    // positions            // normals         // texcoords
    25.0f, -0.5f,  25.0f,  0.0f, 1.0f, 0.0f,  25.0f,  0.0f,
    -25.0f, -0.5f,  25.0f,  0.0f, 1.0f, 0.0f,   0.0f,  0.0f,
    -25.0f, -0.5f, -25.0f,  0.0f, 1.0f, 0.0f,   0.0f, 25.0f,

    25.0f, -0.5f,  25.0f,  0.0f, 1.0f, 0.0f,  25.0f,  0.0f,
    -25.0f, -0.5f, -25.0f,  0.0f, 1.0f, 0.0f,   0.0f, 25.0f,
    25.0f, -0.5f, -25.0f,  0.0f, 1.0f, 0.0f,  25.0f, 25.0f
  };

  glGenVertexArrays(1, VAO);
  glGenBuffers(1, VBO);
  glBindVertexArray(*VAO);
  glBindBuffer(GL_ARRAY_BUFFER, *VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof vertices, vertices, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
  glBindVertexArray(0);
}

void load_quad(unsigned int *VAO, unsigned int *VBO) {
  float vertices[] = {
    // positions        // texture Coords
    -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
    1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
  };

  glGenVertexArrays(1, VAO);
  glGenBuffers(1, VBO);
  glBindVertexArray(*VAO);
  glBindBuffer(GL_ARRAY_BUFFER, *VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
}

unsigned int load_texture(const char* path) {
  unsigned int texture_ID;
  glGenTextures(1, &texture_ID);

  int width, height, channels;
  unsigned char *data = stbi_load(path, &width, &height, &channels, 0);
  if (data) {
    GLenum format;
    switch (channels) {
    case 1: format = GL_RED; break;
    case 3: format = GL_RGB; break;
    case 4: format = GL_RGBA; break;
    default: format = GL_RED; break;
    }

    glBindTexture(GL_TEXTURE_2D, texture_ID);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, format == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT); // for this tutorial: use GL_CLAMP_TO_EDGE to prevent semi-transparent borders. Due to interpolation it takes texels from next repeat 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, format == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  }
  else {
    std::cout << "Texture failed to load at path: " << path << std::endl;
  }
  stbi_image_free(data);
  return texture_ID;
}

unsigned int load_depth_map() {
  unsigned int depth_map;
  glGenTextures(1, &depth_map);
  glBindTexture(GL_TEXTURE_2D, depth_map);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  return depth_map;
}

unsigned int load_framebuffer(unsigned int depth_map) {
  unsigned int FBO;
  glGenFramebuffers(1, &FBO);
  glBindFramebuffer(GL_FRAMEBUFFER, FBO);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_map, 0);
  glDrawBuffer(GL_NONE);
  glReadBuffer(GL_NONE);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  return FBO;
}