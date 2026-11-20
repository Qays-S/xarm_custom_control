#!/usr/bin/env python3
"""
ISOLATED single-call test of the real xArm gripper via GripperCommand.
Does NOT move the arm. Run this alone, watch the physical gripper,
before wiring gripper control into the full sequence.

Run under: ros2 launch xarm_moveit_config _robot_moveit_realmove.launch.py
           robot_ip:=192.168.1.210 dof:=7 robot_type:=xarm add_gripper:=true
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand

GRIPPER_ACTION_REAL = "/xarm_gripper/gripper_action"

# TODO: confirmed range unknown yet. Starting with a modest, conservative
# test value and low effort. DO NOT use large values until range/direction
# is confirmed by watching this run.
TEST_POSITION = 0.3
TEST_MAX_EFFORT = 10.0


def main():
    rclpy.init()
    node = Node("test_real_gripper")

    client = ActionClient(node, GripperCommand, GRIPPER_ACTION_REAL)
    node.get_logger().info("Waiting for gripper action server...")
    client.wait_for_server()

    goal = GripperCommand.Goal()
    goal.command.position = TEST_POSITION
    goal.command.max_effort = TEST_MAX_EFFORT

    node.get_logger().info(
        f"Sending gripper goal: position={TEST_POSITION}, max_effort={TEST_MAX_EFFORT}"
    )
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    goal_handle = send_future.result()

    if not goal_handle.accepted:
        node.get_logger().error("Gripper goal rejected")
        node.destroy_node()
        rclpy.shutdown()
        return

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    node.get_logger().info(f"Gripper movement complete. Result: {result}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()