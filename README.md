# SCAL
ROS implementation of UWB-IMU-Odometer Fusion for Simultaneous Calibration and Localization.
The demonstration of experimental results is as follows:
[demo](https://github.com/JasonSun623/SCAL/blob/main/demo.gif)
# Preparation
## Requirements:
Ubuntu 18.04 ROS melodic

Running this program requires ROS support. If it is not installed, please install ROS first.
## Install dependencies:
sudo apt-get update

sudo apt install python-pip

sudo pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

sudo easy_install -U statsmodels

sudo apt install libcholmod3.0.6 libcsparse3.1.4 libsuitesparse-dev python-cvxopt 

sudo apt install ros-melodic-hector-trajectory-server

git clone https://github.com/RainerKuemmerle/g2o

# Building

Clone the code and compile it as follows

    mkdir  ~/catkin_ws/src
    cd catkin_ws/src
    git clone --recursive https://github.com/JasonSun623/SCAL.git 
    cd ../
    catkin_make
    source devel/setup.bash

# Datasets
#4_anchor.bag

Duration:    57.9s

We adopt a layout design consisting of four anchors and one tag to evaluate SCAL method.

#6_anchor.bag

Duration:    22:25s (1345s)

Topics:      
*  UWB raw measurements
  
   /nodeframe2_rostime : nlink_parser/LinktrackNodeframe2rostime
   
* IMU raw measurements

   /imu : sensor_msgs/Imu                        
            
* Wheel odometry raw measurements

  /odom  : nav_msgs/Odometry

  # Usage
  
Run

    roslaunch nlink_parser linktrack.launch
    rosrun nlink_example linktrack_example
    roslauch localization localization_calibrate.launch
    roslauch localization localization_bag_play.launch
  
