import rclpy
from rclpy.executors import ExternalShutdownException

from xarm_custom_control.move_to_pose import MoveToPose


def main():
    try:
        rclpy.init()
        node = MoveToPose()

        # Same values as move_to_pose.py's near-limit test, kept within
        # MoveIt's enforced +/-178 degree limit on every joint.
        near_limit_pose_deg = [90, 110, 90, 175, 90, 170, 90]
        home_deg = [0, 0, 0, 90, 0, 90, 0]

        # Mirrored version: J1 (base rotation) flipped to the opposite
        # side. J2-J7 stay the same, since those control forward/upward
        # reach rather than left/right side.
        mirrored_pose_deg = [-near_limit_pose_deg[0]] + near_limit_pose_deg[1:]

        node.get_logger().info('Reaching to near-limit pose (side 1)...')
        node.move_to_joints(near_limit_pose_deg)

        node.get_logger().info('Returning to home...')
        node.move_to_joints(home_deg)

        node.get_logger().info('Reaching to near-limit pose (side 2, mirrored)...')
        node.move_to_joints(mirrored_pose_deg)

        node.get_logger().info('Returning to home...')
        node.move_to_joints(home_deg)

        node.get_logger().info('Both-sides reach complete.')

        node.destroy_node()
        rclpy.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
