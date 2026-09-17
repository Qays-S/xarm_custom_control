import rclpy
from xarm_custom_control.move_to_pose import MoveToPose


def main():
    rclpy.init()
    node = MoveToPose()

    y, z = 0.0, 0.25
    x = 0.3           # start from a known-reachable point
    step = 0.02        # step size in meters, moving INWARD toward the base
    min_x = 0.0        # safety floor so this can't loop forever

    last_good_x = None
    found_limit = False

    while x >= min_x and not found_limit:
        node.get_logger().info(f'Trying x = {x:.2f} (moving toward the base)...')
        if node.move_to(x=x, y=y, z=z):
            last_good_x = x
            x -= step
        else:
            node.get_logger().info(f'First failure at x = {x:.2f} (likely self-collision or IK limit)')
            found_limit = True

    node.get_logger().info(
        f'Inward boundary (at y={y}, z={z}): last successful x = {last_good_x}'
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()