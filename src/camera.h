#pragma once

#include <algorithm>
#include <cmath>

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct FlyCamera {
    glm::vec3 position{0.0f, 300.0f, 0.0f};
    float yaw = 0.0f;    // radians, 0 = looking down -Z
    float pitch = 0.1f;  // radians
    float fovY = glm::radians(60.0f);

    glm::vec3 forward() const {
        return {std::cos(pitch) * std::sin(yaw), std::sin(pitch), -std::cos(pitch) * std::cos(yaw)};
    }

    // Reverse-Z (near -> 1, far -> 0) for float depth precision across a 200 km range.
    glm::mat4 viewProj(float aspect) const {
        glm::mat4 view = glm::lookAt(position, position + forward(), glm::vec3(0, 1, 0));
        glm::mat4 proj = glm::perspectiveRH_ZO(fovY, aspect, 200000.0f, 1.0f);
        proj[1][1] *= -1.0f;  // Vulkan clip space has +Y down
        return proj * view;
    }

    // Right mouse drag to look, WASD/QE to move, Shift for speed.
    void update(GLFWwindow* window, float dt, bool mouseCaptured, bool keyboardCaptured) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS && !mouseCaptured && hasLast_) {
            yaw += float(x - lastX_) * 0.003f;
            pitch = std::clamp(pitch - float(y - lastY_) * 0.003f, -1.5f, 1.5f);
        }
        lastX_ = x;
        lastY_ = y;
        hasLast_ = true;
        if (keyboardCaptured) return;

        glm::vec3 f = forward();
        glm::vec3 right = glm::normalize(glm::cross(f, glm::vec3(0, 1, 0)));
        glm::vec3 move{0.0f};
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move += f;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move -= f;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move += right;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move -= right;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) move.y += 1.0f;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) move.y -= 1.0f;
        float speed = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 3000.0f : 300.0f;
        if (glm::length(move) > 0.0f) position += glm::normalize(move) * speed * dt;
    }

private:
    double lastX_ = 0.0, lastY_ = 0.0;
    bool hasLast_ = false;
};
