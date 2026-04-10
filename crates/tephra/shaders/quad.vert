#version 450

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
} pc;

layout(location = 0) out vec2 frag_uv;

void main() {
    gl_Position = pc.mvp * vec4(in_position, 0.0, 1.0);
    frag_uv = in_uv;
}
