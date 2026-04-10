#version 450

layout(set = 0, binding = 0) uniform sampler2D scene_tex;

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

void main() {
    vec2 center = frag_uv - 0.5;
    float dist = length(center);

    // Chromatic aberration: offset R and B channels radially
    float aberration = dist * 0.03;
    float r = texture(scene_tex, frag_uv + center * aberration).r;
    float g = texture(scene_tex, frag_uv).g;
    float b = texture(scene_tex, frag_uv - center * aberration).b;

    // Vignette
    float vignette = 1.0 - dot(center, center) * 2.0;
    vignette = clamp(vignette, 0.0, 1.0);

    out_color = vec4(vec3(r, g, b) * vignette, 1.0);
}
