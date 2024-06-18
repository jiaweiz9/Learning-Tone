#!/usr/bin/env python
"""
This fake listener is used for debugging without connecting to real robot.
"""
import rospy
from std_msgs.msg import Float32MultiArray, Float64MultiArray

def psyonic_fingers_callback(data):
    rospy.loginfo("Psyonic received message: %s", data.data)

def papras_joint6_callback(data):
    rospy.loginfo("PAPRAS Joint6 received message: %s", data.data)

def listener():
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("robot1/psyonic_controller", Float32MultiArray, psyonic_fingers_callback)
    rospy.Subscriber("joint6_controller/command", Float64MultiArray, papras_joint6_callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
