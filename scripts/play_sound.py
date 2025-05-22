#!/usr/bin/python

import rospy
from std_msgs.msg import String
import os

def play_sound(data):
    sound_file = data.data
    file_path = os.path.join("/home/jetson", sound_file)  # Oppdater banen
    os.system("aplay -D hw:2,0 {}".format(file_path))

def listener():
    rospy.init_node('sound_player', anonymous=True)
    rospy.Subscriber('play_sound', String, play_sound)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
