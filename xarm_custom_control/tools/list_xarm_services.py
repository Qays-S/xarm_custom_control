#!/usr/bin/env python3
"""
Run this while xarm7_planner_fake.launch.py is up. It prints every
ROS 2 service whose name looks related to the arm/gripper planners,
along with its service TYPE - which is exactly what you need to know
before writing a client for it.

Usage:
    python3 list_xarm_services.py
"""
import rclpy
from rclpy.node import Node


KEYWORDS = ("plan", "exec", "gripper", "xarm")


def main():
    rclpy.init()
    node = Node("xarm_service_lister")

    # Give discovery a moment to populate
    rclpy.spin_once(node, timeout_sec=1.0)

    services = node.get_service_names_and_types()
    matches = [
        (name, types) for name, types in services
        if any(k in name.lower() for k in KEYWORDS)
    ]

    if not matches:
        print("No matching services found. Is the launch file still running?")
    else:
        print(f"{'SERVICE NAME':45s} TYPE")
        print("-" * 90)
        for name, types in sorted(matches):
            print(f"{name:45s} {', '.join(types)}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
