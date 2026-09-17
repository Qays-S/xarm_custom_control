#!/usr/bin/env python3
"""
lift_to_clearance.py

Moves the xArm7 to the joint configuration that lifts the gripper high
enough to clear the clamps on the table, using the xarm_joint_plan /
xarm_exec_plan services (the reliable path confirmed to work today,
as opposed to /xarm_pose_plan or the raw /xarm/set_servo_angle service).

Joint angles were determined by interactively dragging the arm in RViz
via the MotionPlanning panel's interactive markers, then reading the
resulting joint values off the Joints tab.

NOTE: if running against the fake/sim launch, you may need to loosen
the trajectory controller's goal time tolerance first, e.g.:
    ros2 param set /xarm7_traj_controller constraints.goal_time 5.0
This was needed today because of a controller-manager timing overrun
issue that caused GOAL_TOLERANCE_VIOLATED aborts on the default (much
stricter) tolerance.

Usage:
    ros2 run <your_package> lift_to_clearance.py
    # or directly:
    python3 lift_to_clearance.py
"""

import rclpy
from rclpy.node import Node
from xarm_msgs.srv import PlanJoint, PlanExec


# Joint targets in radians, derived from (degrees):
# J1=2, J2=-84, J3=-6, J4=10, J5=-6, J6=95, J7=0
LIFT_JOINT_TARGET = [0.0349, -1.4661, -0.1047, 0.1745, -0.1047, 1.6580, 0.0]

HOME_JOINT_TARGET = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class LiftToClearance(Node):
    def __init__(self):
        super().__init__('lift_to_clearance')
        self.plan_client = self.create_client(PlanJoint, '/xarm_joint_plan')
        self.exec_client = self.create_client(PlanExec, '/xarm_exec_plan')

        for client, name in [(self.plan_client, '/xarm_joint_plan'),
                              (self.exec_client, '/xarm_exec_plan')]:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for {name} service...')

    def move_to_joint_target(self, target, label):
        self.get_logger().info(f'Planning move to {label}: {target}')
        plan_req = PlanJoint.Request()
        plan_req.target = target

        plan_future = self.plan_client.call_async(plan_req)
        rclpy.spin_until_future_complete(self, plan_future)
        plan_result = plan_future.result()

        if not plan_result.success:
            self.get_logger().error(f'Planning to {label} failed. Aborting.')
            return False

        self.get_logger().info('Plan succeeded. Executing...')
        exec_req = PlanExec.Request()
        exec_req.wait = True

        exec_future = self.exec_client.call_async(exec_req)
        rclpy.spin_until_future_complete(self, exec_future)
        exec_result = exec_future.result()

        if not exec_result.success:
            self.get_logger().error(f'Execution to {label} failed.')
            return False

        self.get_logger().info(f'Successfully moved to {label}.')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = LiftToClearance()

    success = node.move_to_joint_target(LIFT_JOINT_TARGET, 'lift/clearance position')

    if success:
        node.get_logger().info('Arm is now at clamp-clearance height.')
    else:
        node.get_logger().error('Failed to reach clearance position.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
