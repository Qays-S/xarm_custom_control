from setuptools import find_packages, setup

package_name = 'xarm_custom_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qays',
    maintainer_email='qays@todo.todo',
    description='Custom motion control scripts for the xArm7',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'move_to_pose = xarm_custom_control.move_to_pose:main',
            'horizon = xarm_custom_control.horizon:main',
            'reach_both_sides = xarm_custom_control.reach_both_sides:main',
            'wave = xarm_custom_control.wave:main',
            'read_position = xarm_custom_control.read_position:main',
            'square_path = xarm_custom_control.square_path:main',
            'random_reach_test = xarm_custom_control.random_reach_test:main',
            'pick_and_place = xarm_custom_control.pick_and_place:main',
            'interactive_target = xarm_custom_control.interactive_target:main',
            'follow_point = xarm_custom_control.follow_point:main',
            'reachability_sweep = xarm_custom_control.reachability_sweep:main',
            'same_target_variation = xarm_custom_control.same_target_variation:main',
            'find_reach_limit = xarm_custom_control.find_reach_limit:main',
            'orientation_test = xarm_custom_control.orientation_test:main',
            'straight_vs_planned = xarm_custom_control.straight_vs_planned:main',
            'self_collision_boundary = xarm_custom_control.self_collision_boundary:main',
            'lift_to_clearance = xarm_custom_control.lift_to_clearance:main',
            'pixel_picker = xarm_custom_control.pixel_picker:main',
            'wsl_camera_receiver = xarm_custom_control.wsl_camera_receiver:main',
            'list_xarm_services = xarm_custom_control.list_xarm_services:main',
            'xarm_gripper_pick_place_template = xarm_custom_control.xarm_gripper_pick_place_template:main',
            'xarm_waypoint_sequence = xarm_custom_control.xarm_waypoint_sequence:main',
            'test_real_gripper = xarm_custom_control.test_real_gripper:main',
            'xarm_pick_and_place_sequence = xarm_custom_control.xarm_pick_and_place_sequence:main',
        ],
    },
)
