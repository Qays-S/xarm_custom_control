#!/usr/bin/env python3
"""
Pick sequence for the xArm7: approach -> close gripper -> lift -> home.

Arm moves via xarm_planner's plan/exec services. Gripper moves via a
direct FollowJointTrajectory action to its controller (xarm_gripper_planner_node's
own bounds check is broken and silently no-ops, so it's bypassed entirely).

Run under: ros2 launch xarm_planner xarm7_planner_fake.launch.py add_gripper:=true

TODO: PICK_POSITION and LIFT_POSITION below are placeholders. Get real
values by dragging the arm to each pose in RViz (interactive marker),
then reading the seven joint values off the Joints tab (Planning Group:
xarm7). LIFT_POSITION should be the same X/Y as PICK_POSITION but with
the end effector raised clear of the object - not just "one joint tweaked".
"""
import sys

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from xarm_msgs.srv import PlanJoint, PlanExec
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

ARM_JOINT_PLAN_SERVICE = "/xarm_joint_plan"
ARM_EXEC_SERVICE = "/xarm_exec_plan"

GRIPPER_ACTION = "/xarm_gripper_traj_controller/follow_joint_trajectory"
GRIPPER_JOINT_NAME = "drive_joint"

GRIPPER_OPEN = 0.80    # near max (0.855 rad limit) - small safety margin
GRIPPER_CLOSED = 0.02  # near min (0.0 rad limit)

# --- Joint-space targets (radians, one value per joint) - TODO: fill in real values ---
HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PICK_POSITION = [0.5, 0.2, -0.3, 0.4, 0.1, -0.5, 0.2]   # placeholder: at/near the object
LIFT_POSITION = [0.5, 0.0, -0.3, 0.4, 0.1, -0.5, 0.2]   # placeholder: same X/Y, raised clear


def move_arm(node: Node, target: list[float]) -> bool:
    """Plan and execute a joint-space target for the arm via xarm_planner services."""
    plan_client = node.create_client(PlanJoint, ARM_JOINT_PLAN_SERVICE)
    exec_client = node.create_client(PlanExec, ARM_EXEC_SERVICE)

    if not plan_client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(f"Service not available: {ARM_JOINT_PLAN_SERVICE}")
        sys.exit(1)
    if not exec_client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(f"Service not available: {ARM_EXEC_SERVICE}")
        sys.exit(1)

    plan_req = PlanJoint.Request()
    plan_req.target = target
    plan_result = call(node, plan_client, plan_req)
    node.get_logger().info(f"Arm plan: success={plan_result.success}")
    if not plan_result.success:
        return False

    exec_req = PlanExec.Request()
    exec_req.wait = True
    exec_result = call(node, exec_client, exec_req)
    node.get_logger().info(f"Arm exec: success={exec_result.success}")
    return exec_result.success


def move_gripper(node: Node, target_radians: float, seconds: float = 1.0):
    """Send a trajectory goal directly to the gripper's controller."""
    client = ActionClient(node, FollowJointTrajectory, GRIPPER_ACTION)
    node.get_logger().info("Waiting for gripper action server...")
    client.wait_for_server()

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = [GRIPPER_JOINT_NAME]

    point = JointTrajectoryPoint()
    point.positions = [float(target_radians)]
    point.time_from_start = Duration(sec=int(seconds), nanosec=0)
    goal.trajectory.points = [point]

    node.get_logger().info(f"Sending gripper goal: {target_radians} rad")
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    goal_handle = send_future.result()

    if not goal_handle.accepted:
        node.get_logger().error("Gripper goal rejected")
        return

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    node.get_logger().info("Gripper movement complete")


def call(node: Node, client, request):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    return future.result()


def pick_sequence(node: Node):
    """Approach the object, grip it, lift clear, then return home."""
    node.get_logger().info("--- Moving to pick position ---")
    move_arm(node, PICK_POSITION)

    node.get_logger().info("--- Closing gripper ---")
    move_gripper(node, GRIPPER_CLOSED)

    node.get_logger().info("--- Lifting to clearance ---")
    move_arm(node, LIFT_POSITION)

    node.get_logger().info("--- Returning home ---")
    move_arm(node, HOME_POSITION)

    node.get_logger().info("--- Opening gripper ---")
    move_gripper(node, GRIPPER_OPEN)


def main():
    rclpy.init()
    node = Node("xarm_gripper_demo_client")

    pick_sequence(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()