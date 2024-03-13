#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray

def callback(data):
    rospy.loginfo("Received message: %s", data.data)

def listener():
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("robot1/psyonic_controller", Float32MultiArray, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
