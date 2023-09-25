import rospy
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import JointState
from psyonic_hand_control.msg import handVal

        
class FlagSubscriber():
    def __init__(self, default=0):
        self.data = default
        self.flag_sub = rospy.Subscriber('flag', String, self.callback)
    
    def callback(self, data):
        self.data = data

class HandValueSubscriber():
    def __init__(self):
        self.data = []
        self.encoder_sub = rospy.Subscriber('/psyonic_hand_vals', handVal, self.callback)
    def callback(self, data):
        self.data = data.positions

class ForceValueSubscriber():
    def __init__(self):
        self.data = []
        self.encoder_sub = rospy.Subscriber('/psyonic_hand_vals', handVal, self.callback)
    def callback(self, data):
        self.data = data.fingertips

class QPosSubscriber():
    def __init__(self):
        # self.length = 0
        # self.height = 0
        # self.num = 0
        self.data = []
        # self.qpos_sub = rospy.Subscriber('psyonic_controller', Float32MultiArray, self.callback)
        self.qpos_sub = rospy.Subscriber('/joint_states', JointState, self.callback)
        
    
    def callback(self, data):
        # self.length = int(data.layout.dim[0].size)
        # self.height = int(data.layout.dim[1].size)
        # self.num = int(data.layout.dim[2].size)
        self.data = data.position