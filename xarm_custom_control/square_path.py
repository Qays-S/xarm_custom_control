import rclpy
from xarm_custom_control.move_to_pose import MoveToPose


def main():
    rclpy.init()
    node = MoveToPose()

    # Four corners of a flat square, held at a fixed height (z).
    z = 0.25
    corners = [
        (0.25, 0.15, z),
        (0.35, 0.15, z),
        (0.35, -0.15, z),
        (0.25, -0.15, z),
    ]

    for x, y, cz in corners:
        node.get_logger().info(f'Moving to corner ({x}, {y}, {cz})...')
        node.move_to(x=x, y=y, z=cz)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()