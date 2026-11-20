import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateReader(Node):
    def __init__(self):
        super().__init__('joint_state_reader')
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.callback, 10)
        self.received = False

    def callback(self, msg):
        if self.received:
            return
        self.received = True
        self.get_logger().info('Current joint positions (radians):')
        for name, pos in zip(msg.name, msg.position):
            self.get_logger().info(f'  {name}: {pos:.4f}')


def main():
    rclpy.init()
    node = JointStateReader()

    # Spin until one message is received, then stop.
    while rclpy.ok() and not node.received:
        rclpy.spin_once(node, timeout_sec=1.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()