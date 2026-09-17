import rclpy
from xarm_custom_control.move_to_pose import MoveToPose


def main():
    rclpy.init()
    node = MoveToPose()

    # "Pick" location: hover above it, descend, (grip would close here),
    # then lift back up.
    pick_x, pick_y = 0.3, 0.2
    hover_z, grasp_z = 0.3, 0.15

    # "Place" location: same pattern.
    place_x, place_y = 0.3, -0.2

    node.get_logger().info('Hovering above pick location...')
    node.move_to(x=pick_x, y=pick_y, z=hover_z)

    node.get_logger().info('Descending to pick...')
    node.move_to(x=pick_x, y=pick_y, z=grasp_z)

    # NOTE: with a real gripper attached, this is where you'd call a
    # gripper-close command before lifting.

    node.get_logger().info('Lifting...')
    node.move_to(x=pick_x, y=pick_y, z=hover_z)

    node.get_logger().info('Moving to place location...')
    node.move_to(x=place_x, y=place_y, z=hover_z)

    node.get_logger().info('Descending to place...')
    node.move_to(x=place_x, y=place_y, z=grasp_z)

    # NOTE: gripper-open command would go here.

    node.get_logger().info('Lifting...')
    node.move_to(x=place_x, y=place_y, z=hover_z)

    node.get_logger().info('Pick-and-place sequence complete.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()