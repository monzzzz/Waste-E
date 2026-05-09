from __future__ import annotations

import math


def predict(observation, config):
    """Example external policy hook.

    Returns a simple constant-velocity rollout using the observation's current
    speed and yaw rate. Use this as the template for a future Runpod-backed
    model adapter.
    """
    x = 0.0
    y = 0.0
    yaw = 0.0
    speed = max(0.0, float(observation.sample.speed_mps))
    yaw_rate_rad = -math.radians(float(observation.sample.yaw_rate_dps))

    points = []
    for _ in range(config.steps):
        yaw += yaw_rate_rad * config.step_seconds
        x += speed * math.cos(yaw) * config.step_seconds
        y += speed * math.sin(yaw) * config.step_seconds
        points.append([x, y])
    return points
