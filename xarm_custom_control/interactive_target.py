import rclpy
from xarm_custom_control.move_to_pose import MoveToPose


def main():
    rclpy.init()
    node = MoveToPose()

    print("Enter target coordinates as 'x y z' (meters), or 'q' to quit.")

    running = True
    while running:
        raw = input('> ').strip()

        if raw.lower() == 'q':
            running = False
            continue

        parts = raw.split()
        if len(parts) != 3:
            print("Please enter exactly three numbers, e.g. '0.3 0.2 0.25'")
            continue

        try:
            x, y, z = (float(p) for p in parts)
        except ValueError:
            print('Could not parse those as numbers, try again.')
            continue

        node.move_to(x=x, y=y, z=z)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()