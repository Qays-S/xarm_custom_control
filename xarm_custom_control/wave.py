import rclpy
from xarm_custom_control.move_to_pose import MoveToPose


def main():
    rclpy.init()
    node = MoveToPose()

    node.move_to(x=0.3, y=0.2, z=0.3)
    node.move_to(x=0.3, y=-0.2, z=0.3)
    node.move_to(x=0.3, y=0.2, z=0.3)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()