import time

import rclpy
from xarm_custom_control.move_to_pose import MoveToPose


def main():
    rclpy.init()
    node = MoveToPose()

    # Simulated "moving target" -- starts here and drifts a little each step.
    x, y, z = 0.3, -0.2, 0.25
    y_step = 0.04  # how far the target moves per step
    num_steps = 10

    for i in range(num_steps):
        node.get_logger().info(f'Step {i + 1}/{num_steps}: following target to ({x:.2f}, {y:.2f}, {z:.2f})')
        node.move_to(x=x, y=y, z=z)

        y += y_step  # target drifts sideways over time
        time.sleep(0.5)  # pause between steps, as if waiting for a new prediction

    node.get_logger().info('Follow-point demo complete.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()