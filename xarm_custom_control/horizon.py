import math

import rclpy
from rclpy.executors import ExternalShutdownException

from xarm_custom_control.move_to_pose import MoveToPose


def generate_rainbow_waypoints(x, y_center, z_base, radius, num_steps):
    """
    Generates waypoints tracing a semicircular (rainbow) arc in 3D space.

    x: fixed forward distance from the robot base (meters)
    y_center: horizontal center of the arc (meters)
    z_base: height of the arc's two endpoints (meters)
    radius: radius of the arc (meters)
    num_steps: number of waypoints across the 180-degree sweep
    """
    waypoints = []
    for i in range(num_steps):
        theta = math.pi * i / (num_steps - 1)  # 0 -> pi across the sweep

        y = y_center - radius * math.cos(theta)   # side to side
        z = z_base + radius * math.sin(theta)      # low -> peak -> low

        waypoints.append((x, y, z))
    return waypoints


def main():
    try:
        rclpy.init()
        node = MoveToPose()

        # This version uses CARTESIAN pose targets (move_to), not
        # joint-space targets (move_to_joints). That's the key
        # difference: Cartesian planning lets us hold the gripper's
        # orientation fixed for every single waypoint, independent of
        # position -- something joint-space targets can't do, since
        # orientation just falls out of whatever the joint angles
        # happen to produce.
        #
        # (qx=1.0, qy=0.0, qz=0.0, qw=0.0) is the same "facing straight
        # down" orientation confirmed working in move_to_pose.py's
        # original test.
        DOWNWARD_ORIENTATION = dict(qx=1.0, qy=0.0, qz=0.0, qw=0.0)

        x = 0.3
        y_center = 0.0
        z_base = 0.2
        radius = 0.2
        num_steps = 13

        waypoints = generate_rainbow_waypoints(x, y_center, z_base, radius, num_steps)

        node.get_logger().info(
            f'Starting rainbow sweep with fixed downward orientation: '
            f'{num_steps} waypoints.'
        )

        succeeded = 0
        for i, (wx, wy, wz) in enumerate(waypoints):
            node.get_logger().info(f'Waypoint {i + 1}/{num_steps}: ({wx:.2f}, {wy:.2f}, {wz:.2f})')
            success = node.move_to(x=wx, y=wy, z=wz, **DOWNWARD_ORIENTATION)
            if success:
                succeeded += 1
            else:
                node.get_logger().warn(f'Waypoint {i + 1} failed -- skipping to next.')

        node.get_logger().info(
            f'Rainbow sweep complete: {succeeded}/{num_steps} waypoints succeeded.'
        )

        node.destroy_node()
        rclpy.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
