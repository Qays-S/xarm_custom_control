import time

import rclpy
from sensor_msgs.msg import JointState
from xarm_custom_control.move_to_pose import MoveToPose


def get_joint_positions(node, timeout=2.0):
    """One-shot read of the current /joint_states message."""
    result = {}

    def callback(msg):
        result['data'] = list(zip(msg.name, msg.position))

    sub = node.create_subscription(JointState, '/joint_states', callback, 10)

    start = time.time()
    while 'data' not in result and (time.time() - start) < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_subscription(sub)
    return result.get('data')


def main():
    rclpy.init()
    node = MoveToPose()

    # Same Cartesian target sent repeatedly. Since the default planner
    # (OMPL) involves randomized sampling, and inverse kinematics can
    # have multiple valid solutions for one pose, the resulting joint
    # configuration MAY differ between attempts -- though it may also
    # converge to the same solution each time if the IK solver seeds
    # from a similar starting state. This program just shows you
    # what actually happens, rather than assuming either way.
    x, y, z = 0.3, 0.1, 0.25
    num_attempts = 5

    for i in range(num_attempts):
        node.get_logger().info(f'Attempt {i + 1}/{num_attempts}: moving to ({x}, {y}, {z})')
        node.move_to(x=x, y=y, z=z)

        joints = get_joint_positions(node)
        if joints:
            node.get_logger().info(f'Resulting joint positions (radians): {joints}')
        else:
            node.get_logger().warn('Could not read joint state.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()