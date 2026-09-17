#!/usr/bin/env python3
"""
pixel_picker.py

Subscribes to /image_raw, shows the live feed in an OpenCV window,
and prints the pixel (x, y) of wherever you click. Use this during
calibration: move the arm's gripper tip to a table position, place
a small marker there (or use the gripper tip itself if visible),
click on it in this window, and note the printed pixel coordinates.

Press 'q' to quit.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2


class PixelPicker(Node):
    def __init__(self):
        super().__init__('pixel_picker')
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10
        )
        self.latest_frame = None
        cv2.namedWindow('Click to get pixel coords')
        cv2.setMouseCallback('Click to get pixel coords', self.on_click)
        self.get_logger().info('Click on the image to print pixel coordinates. Press q to quit.')

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'Pixel clicked: ({x}, {y})')

    def image_callback(self, msg: Image):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, 3)
        )
        self.latest_frame = img

        cv2.imshow('Click to get pixel coords', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PixelPicker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
