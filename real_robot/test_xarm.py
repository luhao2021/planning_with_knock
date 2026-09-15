"""Manual hardware example. Importing this module does not connect to the robot."""

def main():
    from xarm.wrapper import XArmAPI
    import time
    ROBOT_IP = "192.168.1.231"   # change to your xArm IP

    arm = XArmAPI(ROBOT_IP)

    # Initialize robot
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1)

    # Initialize gripper
    arm.set_gripper_mode(0)
    arm.set_gripper_enable(True)
    arm.set_gripper_speed(3000)


    def move_to_configuration(q, speed=20):
        """q: 7 joint angles in degrees"""
        arm.set_servo_angle(
            angle=q,
            speed=speed,
            wait=True
        )


    def move_to_location(x, y, z, roll, pitch, yaw, speed=100):
        """Position in mm, orientation in degrees"""
        arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=speed,
            wait=True
        )


    def open_gripper():
        arm.set_gripper_position(
            800,
            wait=True
        )


    def close_gripper():
        arm.set_gripper_position(
            100,
            wait=True
        )


    # Example usage
    move_to_configuration([0, -30, 0, 60, 0, 30, 0])

    move_to_location(
        x=400,
        y=0,
        z=300,
        roll=180,
        pitch=0,
        yaw=0
    )

    open_gripper()

    close_gripper()

    arm.disconnect()


if __name__ == "__main__":
    main()
