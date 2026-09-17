#!/usr/bin/env python3
"""
wsl_camera_receiver.py

Runs inside WSL as a ROS 2 node. Listens on a TCP socket for JPEG
frames sent by windows_camera_sender.py (running natively on Windows),
decodes them with OpenCV, and publishes them as sensor_msgs/Image
(encoding bgr8) on /image_raw — a drop-in replacement for usb_cam's
output topic, but without the USB/IP video corruption issues.

Usage:
    python3 wsl_camera_receiver.py

Then start windows_camera_sender.py on the Windows side.
"""

import socket
import struct
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2

HOST = '0.0.0.0'   # listen on all interfaces inside WSL
PORT = 5555


class CameraReceiver(Node):
    def __init__(self):
        super().__init__('wsl_camera_receiver')
        self.publisher = self.create_publisher(Image, '/image_raw', 10)
        self.get_logger().info(
            f'Listening for Windows camera stream on {HOST}:{PORT}, '
            f'publishing to /image_raw'
        )

        # Run the TCP server in a background thread so it doesn't block rclpy.spin()
        self._stop = False
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()

    def _serve(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)

        while not self._stop:
            self.get_logger().info('Waiting for Windows sender to connect...')
            conn, addr = server_sock.accept()
            self.get_logger().info(f'Connected: {addr}')
            try:
                self._handle_connection(conn)
            except (ConnectionResetError, BrokenPipeError):
                self.get_logger().warn('Sender disconnected, waiting for reconnect...')
            finally:
                conn.close()

    def _recv_exact(self, conn, n):
        """Read exactly n bytes from the socket, or return None if the connection closes."""
        buf = b''
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _handle_connection(self, conn):
        while not self._stop:
            header = self._recv_exact(conn, 4)
            if header is None:
                break
            (length,) = struct.unpack('>I', header)

            jpg_bytes = self._recv_exact(conn, length)
            if jpg_bytes is None:
                break

            jpg_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            bgr_img = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
            if bgr_img is None:
                continue

            self._publish(bgr_img)

    def _publish(self, bgr_img):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.height, msg.width = bgr_img.shape[:2]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = bgr_img.tobytes()
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
