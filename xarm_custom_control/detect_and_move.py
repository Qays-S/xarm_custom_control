#!/usr/bin/env python3
"""
Detects the box, converts its position to robot coordinates, and runs a
full pick-and-place: pick up the box from wherever it's detected, and
drop it off at a fixed second location (DROP_X_MM/DROP_Y_MM/DROP_Z_MM).

Each leg of motion (to the box, and to the drop location) uses a
three-stage move to avoid sweeping close to the table:
  1. Lift straight up (same X/Y) to TRAVERSE_Z_MM
  2. Move sideways at TRAVERSE_Z_MM to the new X/Y
  3. Descend straight down to the target height

Full sequence on 'g':
  1. Move to the box (lift -> traverse -> descend to GRIP_Z_MM)
  2. Close gripper
  3. Lift with box to TRAVERSE_Z_MM
  4. Move sideways to the drop location
  5. Descend to DROP_Z_MM
  6. Open gripper
  7. Lift clear to TRAVERSE_Z_MM

SAFETY: this moves the real arm AND operates the real gripper. Only
press 'g' when you're ready to watch the full sequence closely, hand
near the E-stop. Press 'q' to quit without moving.

Run under: ros2 launch xarm_planner xarm7_planner_realmove.launch.py
           robot_ip:=192.168.1.210 kinematics_suffix:=calibrated
"""
import sys
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from xarm_msgs.srv import PlanPose, PlanExec
from xarm_msgs.msg import RobotMsg
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from pixel_to_world import PixelToWorld

# Calibrated HSV range (overhead setup)
HSV_LOWER = np.array([55, 63, 91])
HSV_UPPER = np.array([120, 170, 255])
MIN_CONTOUR_AREA = 500

GRIP_Z_MM = 58.12       # confirmed - height for gripping/releasing the box
TRAVERSE_Z_MM = 114.45  # confirmed - safe height for moving sideways, clears the box

DROP_X_MM = -1.21    # confirmed - second table drop location
DROP_Y_MM = 206.14   # confirmed - second table drop location
DROP_Z_MM = 99.23    # confirmed - second table height (different from pickup table)

GRIPPER_OPEN_REAL = 0.0        # fully open
GRIPPER_GRAB_BOX_REAL = 0.3    # confirmed - grips the box
GRIPPER_ACTION_REAL = "/xarm_gripper/gripper_action"
GRIPPER_MAX_EFFORT = 10.0

# Gripper pointing straight down (roll=pi, pitch=0, yaw=0), consistent with
# every successful pose reading in this project so far.
GRIP_ORIENTATION = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}

ARM_POSE_PLAN_SERVICE = "/xarm_pose_plan"
ARM_EXEC_SERVICE = "/xarm_exec_plan"


def call(node: Node, client, request):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    return future.result()


def move_to_pose(node: Node, plan_client, exec_client, x_mm: float, y_mm: float, z_mm: float) -> bool:
    """Plan and execute a Cartesian pose target. Position in mm, converted to meters here."""
    plan_req = PlanPose.Request()
    plan_req.target.position.x = x_mm / 1000.0
    plan_req.target.position.y = y_mm / 1000.0
    plan_req.target.position.z = z_mm / 1000.0
    plan_req.target.orientation.x = GRIP_ORIENTATION["x"]
    plan_req.target.orientation.y = GRIP_ORIENTATION["y"]
    plan_req.target.orientation.z = GRIP_ORIENTATION["z"]
    plan_req.target.orientation.w = GRIP_ORIENTATION["w"]

    plan_result = call(node, plan_client, plan_req)
    node.get_logger().info(f"Pose plan: success={plan_result.success}")
    if not plan_result.success:
        return False

    exec_req = PlanExec.Request()
    exec_req.wait = True
    exec_result = call(node, exec_client, exec_req)
    node.get_logger().info(f"Pose exec: success={exec_result.success}")
    return exec_result.success


def move_to_target(node: Node, plan_client, exec_client, x_mm: float, y_mm: float,
                    current_x_mm: float, current_y_mm: float, target_z_mm: float = GRIP_Z_MM) -> bool:
    """Three-stage move: lift -> traverse sideways -> descend, avoiding a low sweep."""
    node.get_logger().info(f"Stage 1/3: lift to traverse height ({TRAVERSE_Z_MM} mm) at current X/Y")
    if not move_to_pose(node, plan_client, exec_client, current_x_mm, current_y_mm, TRAVERSE_Z_MM):
        node.get_logger().error("Stage 1 (lift) failed.")
        return False

    node.get_logger().info(f"Stage 2/3: move sideways to target X/Y at traverse height")
    if not move_to_pose(node, plan_client, exec_client, x_mm, y_mm, TRAVERSE_Z_MM):
        node.get_logger().error("Stage 2 (traverse) failed.")
        return False

    node.get_logger().info(f"Stage 3/3: descend to target height ({target_z_mm} mm)")
    if not move_to_pose(node, plan_client, exec_client, x_mm, y_mm, target_z_mm):
        node.get_logger().error("Stage 3 (descend) failed.")
        return False

    return True


def move_gripper(node: Node, position: float, max_effort: float = GRIPPER_MAX_EFFORT):
    """Send a GripperCommand goal directly to the real gripper action server."""
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


def pick_and_place(node: Node, plan_client, exec_client, pick_x: float, pick_y: float,
                    current_x_mm: float, current_y_mm: float) -> tuple[bool, float, float]:
    """Full pick-and-place: go to box, grip, lift, move to drop, release, lift.
    Returns (success, new_current_x, new_current_y)."""

    node.get_logger().info("=== PICK: moving to box ===")
    if not move_to_target(node, plan_client, exec_client, pick_x, pick_y,
                           current_x_mm, current_y_mm, target_z_mm=GRIP_Z_MM):
        return False, pick_x, pick_y

    node.get_logger().info("=== Closing gripper on box ===")
    move_gripper(node, GRIPPER_GRAB_BOX_REAL)

    node.get_logger().info("=== Lifting with box ===")
    if not move_to_pose(node, plan_client, exec_client, pick_x, pick_y, TRAVERSE_Z_MM):
        node.get_logger().error("Lift after grip failed.")
        return False, pick_x, pick_y

    node.get_logger().info("=== PLACE: moving to drop location ===")
    if not move_to_pose(node, plan_client, exec_client, DROP_X_MM, DROP_Y_MM, TRAVERSE_Z_MM):
        node.get_logger().error("Traverse to drop location failed.")
        return False, pick_x, TRAVERSE_Z_MM

    node.get_logger().info("=== Descending to release ===")
    if not move_to_pose(node, plan_client, exec_client, DROP_X_MM, DROP_Y_MM, DROP_Z_MM):
        node.get_logger().error("Descend at drop location failed.")
        return False, DROP_X_MM, DROP_Y_MM

    node.get_logger().info("=== Opening gripper to release ===")
    move_gripper(node, GRIPPER_OPEN_REAL)

    node.get_logger().info("=== Lifting clear after release ===")
    if not move_to_pose(node, plan_client, exec_client, DROP_X_MM, DROP_Y_MM, TRAVERSE_Z_MM):
        node.get_logger().error("Final lift failed.")
        return False, DROP_X_MM, DROP_Y_MM

    return True, DROP_X_MM, DROP_Y_MM


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera at index {camera_index}.")
        sys.exit(1)

    try:
        converter = PixelToWorld("camera_calibration.json")
    except FileNotFoundError:
        print("camera_calibration.json not found. Run calibrate_camera.py first.")
        sys.exit(1)

    rclpy.init()
    node = Node("detect_and_move")
    plan_client = node.create_client(PlanPose, ARM_POSE_PLAN_SERVICE)
    exec_client = node.create_client(PlanExec, ARM_EXEC_SERVICE)

    if not plan_client.wait_for_service(timeout_sec=5.0):
        print(f"Service not available: {ARM_POSE_PLAN_SERVICE}")
        sys.exit(1)
    if not exec_client.wait_for_service(timeout_sec=5.0):
        print(f"Service not available: {ARM_EXEC_SERVICE}")
        sys.exit(1)

    # Get the arm's real current X/Y at startup, so the very first move
    # lifts from its TRUE current position rather than guessing.
    current_pose = {}

    def robot_states_callback(msg):
        current_pose["x"] = msg.pose[0]
        current_pose["y"] = msg.pose[1]

    states_sub = node.create_subscription(RobotMsg, "/xarm/robot_states", robot_states_callback, 10)
    print("Reading current arm position...")
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.1)
        if "x" in current_pose:
            break

    if "x" not in current_pose:
        print("Could not read current robot position - check robot_states is publishing.")
        sys.exit(1)

    current_x_mm = current_pose["x"]
    current_y_mm = current_pose["y"]
    print(f"Starting position: X={current_x_mm:.2f} mm, Y={current_y_mm:.2f} mm")
    node.destroy_subscription(states_sub)

    print(f"GRIP_Z_MM={GRIP_Z_MM}, TRAVERSE_Z_MM={TRAVERSE_Z_MM}")
    print(f"DROP_X_MM={DROP_X_MM}, DROP_Y_MM={DROP_Y_MM}, DROP_Z_MM={DROP_Z_MM}")
    print("Press 'g' to run a full pick-and-place on the detected box. Press 'q' to quit.\n")

    last_world_pos = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            display = frame.copy()
            last_world_pos = None

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area > MIN_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(largest)
                    cx, cy = x + w // 2, y + h // 2

                    world_x, world_y = converter.convert(cx, cy)
                    last_world_pos = (world_x, world_y)

                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(
                        display, f"world=({world_x:.1f}, {world_y:.1f}) mm",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    )

            cv2.imshow("Detect and Move", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("g"):
                if last_world_pos is None:
                    print("No box detected right now - can't move.")
                    continue

                wx, wy = last_world_pos

                print(f"\nStarting pick-and-place: box at X={wx:.2f} mm, Y={wy:.2f} mm, drop at X={DROP_X_MM} mm, Y={DROP_Y_MM} mm...")
                success, current_x_mm, current_y_mm = pick_and_place(
                    node, plan_client, exec_client, wx, wy, current_x_mm, current_y_mm
                )
                print(f"Pick-and-place {'succeeded' if success else 'FAILED'}.\n")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()