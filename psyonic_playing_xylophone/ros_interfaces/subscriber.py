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


class ForceValueSubscriber():
    def __init__(self):
        self.data = []
        self.encoder_sub = rospy.Subscriber('/robot1/psyonic_hand_vals', handVal, self.callback)
    def callback(self, data):
        self.data = data.fingertips

class QPosSubscriber():
    def __init__(self):
        # self.length = 0
        # self.height = 0
        # self.num = 0
        self.data = []
        self.qpos_sub = rospy.Subscriber('psyonic_controller', Float32MultiArray, self.callback)
        
    
    def callback(self, data):
        # self.length = int(data.layout.dim[0].size)
        # self.height = int(data.layout.dim[1].size)
        # self.num = int(data.layout.dim[2].size)
        self.data = data.positions[-1]



class HandValueSubscriber():
    def __init__(self):
        self.data = 0
        self.hand_thumb_sub = rospy.Subscriber('/robot1/psyonic_hand_vals', handVal, self.callback)
    def callback(self, data):
        # print("IN CALLBACK")
        self.data = data.positions[-1]


class PAPRASJoint6PosSubscriber():
    def __init__(self) -> None:
        self.data = 0
        self.qpos_sub = rospy.Subscriber('/joint_states', JointState, self.callback)

    def callback(self, data):
        self.data = data.position[-1]
# rospy.init_node('psyonic', anonymous=True)
# val_subscriber = HandValueSubscriber()
# while True:
#     print(val_subscriber.data)
#     rospy.sleep(0.2)
#     rospy.spin()