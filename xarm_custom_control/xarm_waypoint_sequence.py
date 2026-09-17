#!/usr/bin/env python3
"""
Move the xArm7 through a fixed sequence of joint-space waypoints,
with real gripper actions interleaved via GripperCommand.

Run under: ros2 launch xarm_moveit_config _robot_moveit_realmove.launch.py
           robot_ip:=192.168.1.210 dof:=7 robot_type:=xarm add_gripper:=true

Gripper direction (confirmed on real hardware): 0.0 = fully open,
~0.855 = fully closed. 0.3 confirmed to grip the box.
"""
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from xarm_msgs.srv import PlanJoint, PlanExec
from control_msgs.action import GripperCommand
from controller_manager_msgs.srv import ListControllers, SwitchController

ARM_JOINT_PLAN_SERVICE = "/xarm_joint_plan"
ARM_EXEC_SERVICE = "/xarm_exec_plan"

GRIPPER_ACTION_REAL = "/xarm_gripper/gripper_action"
GRIPPER_MAX_EFFORT = 10.0

GRIPPER_OPEN_REAL = 0.0        # fully open
GRIPPER_GRAB_BOX_REAL = 0.3    # confirmed - grips the box
GRIPPER_CLOSED_REAL = 0.855    # fully closed (fingers together, no object)

TRAJ_CONTROLLER_NAME = "xarm7_traj_controller"
LIST_CONTROLLERS_SERVICE = "/controller_manager/list_controllers"
SWITCH_CONTROLLER_SERVICE = "/controller_manager/switch_controller"

SEQUENCE = [
    ("gripper", GRIPPER_OPEN_REAL),                                        # ensure known starting state
    ("arm", [0.0, -1.4661, 0.0, 0.1745, 0.0, 1.6580, 0.0]),                # stop 1
    ("arm", [-1.5534, -1.4661, 0.0, 0.1745, 0.0, 1.6580, 0.0]),            # stop 2
    ("arm", [-1.5534, -0.5585, 0.0, 0.1745, 0.0, 0.7330, 0.0]),            # stop 3
    ("gripper", GRIPPER_GRAB_BOX_REAL),                                    # grip the box here
    ("arm", [-1.5534, -0.7854, 0.0, 0.7330, 0.0, 1.5184, 0.0]),            # stop 4
    ("arm", [-1.5534, 0.2094, 0.0, 1.6406, 0.0, 1.4312, 0.0]),             # new position (drop)
    ("arm", [-1.5534, 0.3665, 0.0, 1.2217, 0.0, 0.8378, 0.0]),             # drop position
    ("gripper", GRIPPER_OPEN_REAL),                                        # release object
    ("arm", [-1.5534, -0.7854, 0.0, 0.7330, 0.0, 1.5184, 0.0]),            # back through stop 4
    ("arm", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),                          # home
]


def ensure_controller_active(node: Node, controller_name: str = TRAJ_CONTROLLER_NAME) -> bool:
    """Check whether the trajectory controller is active, and reactivate it if not.

    This guards against a recurring issue where xarm7_traj_controller ends up
    'inactive' between runs (e.g. after an E-stop, daemon reset, or unrelated
    controller-manager activity), which otherwise causes every arm exec to
    fail with 'Goal was rejected by server'.
    """
    list_client = node.create_client(ListControllers, LIST_CONTROLLERS_SERVICE)
    if not list_client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(f"Service not available: {LIST_CONTROLLERS_SERVICE}")
        return False

    list_result = call(node, list_client, ListControllers.Request())
    matches = [c for c in list_result.controller if c.name == controller_name]

    if not matches:
        node.get_logger().error(f"Controller '{controller_name}' not found in controller_manager")
        return False

    state = matches[0].state
    node.get_logger().info(f"{controller_name} state: {state}")

    if state == "active":
        return True

    node.get_logger().warn(f"{controller_name} is '{state}', attempting to activate...")

    switch_client = node.create_client(SwitchController, SWITCH_CONTROLLER_SERVICE)
    if not switch_client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(f"Service not available: {SWITCH_CONTROLLER_SERVICE}")
        return False

    switch_req = SwitchController.Request()
    switch_req.activate_controllers = [controller_name]
    switch_req.deactivate_controllers = []
    switch_req.strictness = SwitchController.Request.BEST_EFFORT
    switch_result = call(node, switch_client, switch_req)

    if switch_result.ok:
        node.get_logger().info(f"{controller_name} activated successfully")
        return True
    else:
        node.get_logger().error(f"Failed to activate {controller_name}")
        return False


def move_arm(node: Node, plan_client, exec_client, target: list[float]) -> bool:
    """Plan and execute a joint-space target for the arm."""
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


def move_gripper(node: Node, position: float, max_effort: float = GRIPPER_MAX_EFFORT):
    """Send a GripperCommand goal directly to the real gripper's action server."""
    client = ActionClient(node, GripperCommand, GRIPPER_ACTION_REAL)
    client.wait_for_server()

    goal = GripperCommand.Goal()
    goal.command.position = position
    goal.command.max_effort = max_effort

    node.get_logger().info(f"Gripper target: {position}")
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


def main():
    rclpy.init()
    node = Node("xarm_waypoint_sequence")

    plan_client = node.create_client(PlanJoint, ARM_JOINT_PLAN_SERVICE)
    exec_client = node.create_client(PlanExec, ARM_EXEC_SERVICE)

    if not plan_client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(f"Service not available: {ARM_JOINT_PLAN_SERVICE}")
        sys.exit(1)
    if not exec_client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(f"Service not available: {ARM_EXEC_SERVICE}")
        sys.exit(1)

    if not ensure_controller_active(node):
        node.get_logger().error("Could not confirm trajectory controller is active. Aborting.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    sequence_ok = True

for i, (kind, value) in enumerate(SEQUENCE, start=1):
    if not sequence_ok:
        continue

    time.sleep(1.5)  # let the previous motion fully settle before the next command

    node.get_logger().info(f"--- Step {i} of {len(SEQUENCE)}: {kind} ---")
    if kind == "arm":
        success = move_arm(node, plan_client, exec_client, value)
        if not success:
            node.get_logger().error(f"Step {i} failed, stopping sequence.")
            sequence_ok = False
    elif kind == "gripper":
        move_gripper(node, value)
    else:
        node.get_logger().error(f"Unknown step type: {kind}")
        sequence_ok = False

        time.sleep(0.5)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()