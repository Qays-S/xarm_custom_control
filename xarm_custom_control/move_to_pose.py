import math

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from xarm_msgs.srv import PlanPose, PlanJoint, PlanExec
from geometry_msgs.msg import Pose


class MoveToPose(Node):

    def __init__(self):
        super().__init__('move_to_pose')

        self.pose_plan_client = self.create_client(PlanPose, '/xarm_pose_plan')
        self.joint_plan_client = self.create_client(PlanJoint, '/xarm_joint_plan')
        self.exec_client = self.create_client(PlanExec, '/xarm_exec_plan')

        while not self.pose_plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /xarm_pose_plan service...')

        while not self.joint_plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /xarm_joint_plan service...')

        while not self.exec_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /xarm_exec_plan service...')

    def _execute_last_plan(self):
        exec_req = PlanExec.Request()
        exec_req.wait = True

        exec_future = self.exec_client.call_async(exec_req)
        rclpy.spin_until_future_complete(self, exec_future)
        exec_result = exec_future.result()

        if exec_result is None or not exec_result.success:
            self.get_logger().error('Execution failed.')
            return False

        self.get_logger().info('Execution succeeded.')
        return True

    def move_to(self, x, y, z, qx=1.0, qy=0.0, qz=0.0, qw=0.0):
        target = Pose()
        target.position.x = x
        target.position.y = y
        target.position.z = z
        target.orientation.x = qx
        target.orientation.y = qy
        target.orientation.z = qz
        target.orientation.w = qw

        plan_req = PlanPose.Request()
        plan_req.target = target

        self.get_logger().info(f'Requesting pose plan to ({x}, {y}, {z})...')
        plan_future = self.pose_plan_client.call_async(plan_req)
        rclpy.spin_until_future_complete(self, plan_future)
        plan_result = plan_future.result()

        if plan_result is None or not plan_result.success:
            self.get_logger().error('Pose planning failed.')
            return False

        self.get_logger().info('Plan succeeded. Executing...')
        return self._execute_last_plan()

    def move_to_joints(self, joint_angles_deg):
        """
        joint_angles_deg: list of 7 angles in DEGREES, one per joint (J1..J7).
        Converted to radians internally, since xarm_msgs expects radians.
        """
        joint_angles_rad = [math.radians(a) for a in joint_angles_deg]

        plan_req = PlanJoint.Request()
        plan_req.target = joint_angles_rad

        self.get_logger().info(f'Requesting joint plan to {joint_angles_deg} degrees...')
        plan_future = self.joint_plan_client.call_async(plan_req)
        rclpy.spin_until_future_complete(self, plan_future)
        plan_result = plan_future.result()

        if plan_result is None or not plan_result.success:
            self.get_logger().error('Joint planning failed (likely outside reachable limits).')
            return False

        self.get_logger().info('Plan succeeded. Executing...')
        return self._execute_last_plan()


def main():
    try:
        rclpy.init()
        node = MoveToPose()

        # xArm7 official joint ranges (degrees), per UFACTORY spec:
        # J1: +/-360   J2: -117 to 116   J3: +/-360
        # J4: -6 to 225   J5: +/-360   J6: -97 to 180   J7: +/-360
        #
        # NOTE: Full range values (e.g. J1=360) are usually NOT reachable in
        # one straight-line joint plan due to self-collision / cable limits
        # enforced by the planner, even though the joint itself allows it.
        # These are deliberately conservative "near-limit" test values,
        # safe to try first before pushing closer to the true extremes.
        near_limit_pose_deg = [90, 110, 90, 200, 90, 170, 90]

        node.move_to_joints(near_limit_pose_deg)

        # Return to a safe, neutral "home" position afterward.
        home_deg = [0, 0, 0, 90, 0, 90, 0]
        node.move_to_joints(home_deg)

        node.destroy_node()
        rclpy.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
