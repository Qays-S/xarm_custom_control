import random

import rclpy
from xarm_custom_control.move_to_pose import MoveToPose


def main():
    rclpy.init()
    node = MoveToPose()

    num_attempts = 10
    # Safe-ish bounds based on values that have worked reliably so far.
    x_range = (0.25, 0.35)
    y_range = (-0.25, 0.25)
    z_range = (0.15, 0.35)

    successes = 0
    for i in range(num_attempts):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        z = random.uniform(*z_range)

        node.get_logger().info(f'Attempt {i + 1}/{num_attempts}: ({x:.2f}, {y:.2f}, {z:.2f})')
        if node.move_to(x=x, y=y, z=z):
            successes += 1

    node.get_logger().info(f'Random reach test complete: {successes}/{num_attempts} succeeded.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()