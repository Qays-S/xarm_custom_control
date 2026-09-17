import rclpy
from xarm_custom_control.move_to_pose import MoveToPose


def main():
    rclpy.init()
    node = MoveToPose()

    # (x, y, z) targets to try in order, top to bottom.
    targets = [
        # Comfortable, known-good range
        (0.3, 0.0, 0.25),
        (0.3, 0.2, 0.25),
        (0.3, -0.2, 0.25),
        (0.25, 0.15, 0.3),

        # Further out horizontally
        (0.45, 0.0, 0.25),
        (0.5, 0.3, 0.2),
        (0.4, -0.35, 0.25),

        # Higher and lower
        (0.3, 0.0, 0.5),
        (0.3, 0.0, 0.6),
        (0.3, 0.0, 0.05),
    ]

    successes = 0
    for i, (x, y, z) in enumerate(targets):
        node.get_logger().info(f'Target {i + 1}/{len(targets)}: ({x}, {y}, {z})')
        if node.move_to(x=x, y=y, z=z):
            successes += 1
        else:
            node.get_logger().warn(f'Target {i + 1} failed -- moving on.')

    node.get_logger().info(f'Sweep complete: {successes}/{len(targets)} targets succeeded.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()