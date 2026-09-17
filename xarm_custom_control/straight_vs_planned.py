import rclpy
from xarm_msgs.srv import PlanSingleStraight
from geometry_msgs.msg import Pose
from xarm_custom_control.move_to_pose import MoveToPose


def main():
    rclpy.init()
    node = MoveToPose()

    # NOTE: xarm_msgs' PlanSingleStraight service is assumed here to use
    # a 'target' field (a single geometry_msgs/Pose), matching the
    # pattern used by PlanPose. If this errors, check the real field
    # name with:  ros2 interface show xarm_msgs/srv/PlanSingleStraight
    node.straight_plan_client = node.create_client(PlanSingleStraight, '/xarm_straight_plan')
    while not node.straight_plan_client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for /xarm_straight_plan service...')

    target = Pose()
    target.position.x = 0.3
    target.position.y = 0.2
    target.position.z = 0.3
    target.orientation.x = 1.0
    target.orientation.w = 0.0

    # Return to a neutral spot first, so both moves start from the same place.
    node.get_logger().info('Returning to a neutral start position...')
    node.move_to(x=0.3, y=0.0, z=0.25)

    node.get_logger().info('Move 1: planned (OMPL) path to target...')
    node.move_to(x=target.position.x, y=target.position.y, z=target.position.z)

    node.get_logger().info('Returning to neutral again...')
    node.move_to(x=0.3, y=0.0, z=0.25)

    node.get_logger().info('Move 2: straight-line path to the SAME target...')
    req = PlanSingleStraight.Request()
    req.target = target

    plan_future = node.straight_plan_client.call_async(req)
    rclpy.spin_until_future_complete(node, plan_future)
    result = plan_future.result()
    if result is not None and result.success:
        node.get_logger().info('Straight-line plan succeeded. Executing...')
        node._execute_last_plan()
    else:
        node.get_logger().error('Straight-line plan failed.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()