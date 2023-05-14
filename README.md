# qt_robot

# videotester.py
You can use the `requirements.txt` to install packages. It should have the following package versions. To check your package versions use `pip freeze`
```
Python 3.10.6
tensorflow: 2.12.0
keras: 2.12.0
cv2: 4.7.0.72
numpy: 1.23.5

python3 -m pip install -r requirements.txt
```

To run on command line:
```
python3 videotester.py
```


# QT Robot Setup:
Initial QT robot set up followed: https://docs.luxai.com/docs/v1/intro_code
```
Version: QTrobot V1 (older than QTRDTP2105!)

# To connect from your laptop/PC you can connect to wifi and set the following in your ~/.bash_aliases
wlan0: 192.168.178.62

source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

## QTrobot
export ROS_IP=192.168.178.62
export ROS_MASTER_URI=http://192.168.100.1:11311    
```

# QT Robot tutorial
To build a single package (e.g. my_tutorial) from source directory, you can call `catkin_make --pkg my_tutorial`! 


