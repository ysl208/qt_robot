#!/usr/bin/env python
import sys
import rospy
from std_msgs.msg import String

if __name__ == '__main__':
    rospy.init_node('my_tutorial_node')
    rospy.loginfo("my_tutorial_node started!")

    # creating a ros publisher
    speechSay_pub = rospy.Publisher('/qt_robot/behavior/talkText', String, queue_size=10)
    # wait for publisher/subscriber connections
    wtime_begin = rospy.get_time()
    while (speechSay_pub.get_num_connections()) :
        rospy.loginfo("waiting for subscriber connections")
        if rospy.get_time() - wtime_begin > 5.0:
            rospy.logerr("Timeout while waiting for subscribers connection!")
            sys.exit()
        rospy.sleep(0.5)

    # publish a text message to TTS
    speechSay_pub.publish("Hello! my name is QT!")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

    rospy.loginfo("finished!")

