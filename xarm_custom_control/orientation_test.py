import math

import rclpy
from xarm_custom_control.move_to_pose import MoveToPose


def quat_multiply(q1, q2):
    """Multiply two quaternions, each as (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def axis_angle_to_quat(axis, angle_rad):
    """axis: (x, y, z) unit vector. Returns (x, y, z, w)."""
    s = math.sin(angle_rad / 2)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle_rad / 2))


def main():
    rclpy.init()
    node = MoveToPose()

    x, y, z = 0.3, 0.0, 0.25

    # Base orientation: gripper facing straight down.
    base_down = (1.0, 0.0, 0.0, 0.0)

    # A few small tilts (degrees) away from straight-down, rotating
    # about the Y axis (pitch).
    tilt_degrees = [0, 10, 20, 30, 45]

    for deg in tilt_degrees:
        tilt_quat = axis_angle_to_quat((0, 1, 0), math.radians(deg))
        combined = quat_multiply(base_down, tilt_quat)

        node.get_logger().info(f'Trying {deg} degree tilt at ({x}, {y}, {z})...')
        success = node.move_to(x=x, y=y, z=z, qx=combined[0], qy=combined[1], qz=combined[2], qw=combined[3])
        node.get_logger().info(f'  {deg} degrees: {"OK" if success else "FAILED"}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()