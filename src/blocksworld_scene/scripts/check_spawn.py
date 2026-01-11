#!/usr/bin/env python3
# check_spawn_entity.py
#
# Verify that /spawn_entity exists *and* speaks gazebo_msgs/srv/SpawnEntity.
#
# Run:
#   source ~/ros2_ws/install/setup.bash
#   python3 check_spawn_entity.py
#
# If the service is ready you should see:
#   ➜  Service /spawn_entity READY                    (in ~X ms)
#   ➜  Call returned success = True                  (or the error message)

import sys, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity

TIMEOUT_SEC = 5.0          # how long we wait for the service

class SpawnTester(Node):
    def __init__(self):
        super().__init__('spawn_entity_tester')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')

    def wait_and_report(self):
        start = time.time()
        while not self.cli.wait_for_service(timeout_sec=0.1):
            if time.time() - start > TIMEOUT_SEC:
                self.get_logger().error('Service /spawn_entity NOT available.')
                return False
        elapsed = int((time.time()-start)*1000)
        self.get_logger().info(f'Service /spawn_entity READY  (in ~{elapsed} ms)')
        return True

    def ping_service(self):
        req = SpawnEntity.Request()
        req.name            = 'ping_block'
        req.xml             = '<sdf version="1.7"><model name="ping_block"><static>true</static></model></sdf>'
        req.robot_namespace = ''
        req.initial_pose    = Pose()          # all zeros → origin
        req.reference_frame = 'world'
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.done():
            self.get_logger().info(f'Call returned success = {future.result().success}')
            self.get_logger().info(f'status_message = "{future.result().status_message}"')
        else:
            self.get_logger().error('No reply (timed out).')

def main():
    rclpy.init()
    tester = SpawnTester()
    if tester.wait_and_report():
        tester.ping_service()
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

