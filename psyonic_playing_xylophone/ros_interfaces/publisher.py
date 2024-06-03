import rospy, sys, os
import numpy as np
from std_msgs.msg import Float32MultiArray, String
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from psyonic_playing_xylophone.ros_interfaces.ros_np_multiarray import ros_np_multiarray as rnm
from sensor_msgs.msg import JointState

class FlagPublisher():
    def __init__(self):
        self.flag_pub = rospy.Publisher('flag', String, queue_size=10)
        
    def run(self, data):
        data = "{}".format(data)
        self.flag_pub.publish(data)
    
    def publish_after_delay(self, data, delay=0.02):
        rospy.sleep(delay)
        self.publish_once(data)

    def publish_once(self, data):
        while not rospy.is_shutdown():
            # rate = rospy.Rate(50)
            connections = self.flag_pub.get_num_connections()
            if connections > 0:
                self.run(data)
                # rospy.loginfo(f'{data} Flag published')
                # rate.sleep()
                break

class HandValuePublisher():
    def __init__(self):
        self.encoder_pub = rospy.Publisher('robot1/psyonic_hand_vals', Float32MultiArray, queue_size=10)
    
    def run(self, data):
        self.encoder_pub.publish(data)
    
    def publish_once(self, data):
        data = rnm.to_multiarray_f32(data)
        while not rospy.is_shutdown():
            # rate = rospy.Rate(30)
            connections = self.encoder_pub.get_num_connections()
            if connections > 0:
                self.run(data)
                rospy.loginfo('Encoder value published')
                break



class JointStatePublisher():
    def __init__(self):
        # self.qpos_pub = rospy.Publisher('psyonic_controller', Float32MultiArray, queue_size=10)
        self.qpos_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    def run(self, data):
        self.qpos_pub.publish(data)
    
    def publish_once(self, data):
        # data = rnm.to_multiarray_f32(data)
        while not rospy.is_shutdown():
            rate = rospy.Rate(30)
            connections = self.qpos_pub.get_num_connections()
            if connections > 0:
                self.run(data)
                rospy.loginfo('Qpos published')
                rate.sleep()
                break

class SoundPublisher():
    def __init__(self):
        self.sound_pub = rospy.Publisher('/sound_plot', Float32MultiArray, queue_size=1000)
    def run(self, data):
        self.sound_pub.publish(data)
    def publish_once(self, data):
        data = rnm.to_multiarray_f32(data)
        while not rospy.is_shutdown():
            # rate = rospy.Rate(50) # 50HZ
            connections = self.sound_pub.get_num_connections()
            # print(connections)
            if connections > 0:
                self.run(data)
                # rospy.loginfo('Qpos published')
                # rate.sleep()
                break


class QPosPublisher():
    def __init__(self):
        self.qpos_pub = rospy.Publisher('robot1/psyonic_controller', Float32MultiArray, queue_size=1000)\
    
    def run(self, data):
        self.qpos_pub.publish(data)
    
    def publish_once(self, data):
        data = rnm.to_multiarray_f32(data)
        while not rospy.is_shutdown():
            rate = rospy.Rate(50) # 50HZ
            connections = self.qpos_pub.get_num_connections()
            # print(connections)
            if connections > 0:
                self.run(data)
                # rospy.loginfo('Qpos published')
                rate.sleep()
                break
    