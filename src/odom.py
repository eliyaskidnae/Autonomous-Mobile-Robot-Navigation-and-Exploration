#!/usr/bin/python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf
import math
from std_msgs.msg import Float64MultiArray
from math import atan2 , sqrt , degrees , radians , pi , floor , cos , sin
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2
from std_msgs.msg import ColorRGBA
import threading

class Odom:
    def __init__(self) -> None:

        #Initilaize the state of the robot
        self.xk = np.array([0, 0, 0]).reshape(3, 1) # Initialize the state of the robot
        self.xk = np.array([3., -0.78, np.pi/2]).reshape(3, 1) # Initialize the state of the robot
        self.Pk = np.eye(3)*0.000 # Initialize the covariance of the state
        self.map =  [] # Initialize the map
        self.scan = [] # Initialize the scan
        self.Qk= np.array([ [0.2,0 ],
                            [0, 0.2],
                           ]) 

        # Initialize the velocities
        self.parent_frame     = "world_ned"
        self.child_frame      = "turtlebot/base_footprint"
        # self.child_frame  = "turtlebot/kobuki/base_footprint"

        self.wheel_name_left  = "turtlebot/wheel_left_joint"
        self.wheel_name_right = "turtlebot/wheel_right_joint"

        self.left_wheel_velocity   = 0
        self.right_wheel_velocity  = 0
        self.left_wheel_velo_read  = False
        self.right_wheel_velo_read = False

        self.last_time    = rospy.Time.now().to_sec()
        self.wheel_radius = 0.035
        self.wheel_base   = 0.235 
        self.scan_th_distance = 0.8
        self.scan_th_angle    = np.pi/2
        self.mutex            = threading.Lock()
        #creare a publisher
        self.tf_broadcaster = tf.TransformBroadcaster()
         # Create a publisher
        # odom publisher
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)
        self.vel_pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray , queue_size=10)
        self.vel_pub = rospy.Publisher('/mobile_base/commands/wheel_velocities', Float64MultiArray , queue_size=10)
      
        # joint state subscriber
        rospy.Subscriber('/turtlebot/joint_states', JointState, self.joint_state_callback)
        rospy.Subscriber('/kobuki/joint_states', JointState, self.joint_state_callback)
        rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        rospy.Subscriber('/turtlebot/kobuki/sensors/imu_data', Imu, self.imu_callback)
        rospy.Subscriber('/kobuki/kobuki/sensors/imu_data', Imu, self.imu_callback)
       
    def wrap_angle(self, angle):
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    
    def gt_odom_callback(self, msg):
        odom = Odometry()
        
        pose = msg.pose.pose    
        orientation = pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.gt_xk = np.array([pose.position.x, pose.position.y, yaw]).reshape(3, 1)
#
    def joint_state_callback(self, msg):
        self.mutex.acquire()
        if(msg.name[0]== self.wheel_name_left):        
            self.left_wheel_velocity = msg.velocity[0]
            self.left_wheel_velo_read = True                   
        elif(msg.name[0] == self.wheel_name_right):

            self.right_wheel_velocity = msg.velocity[0]
            self.right_wheel_velo_read = True
                 
            if(self.left_wheel_velo_read ):
               
                self.left_wheel_velo_read = False
                self.right_wheel_velo_read = False
                self.left_linear_vel = self.left_wheel_velocity * self.wheel_radius
                self.right_linear_vel = self.right_wheel_velocity * self.wheel_radius 
                self.v = (self.left_linear_vel + self.right_linear_vel) / 2
                self.w = (self.left_linear_vel - self.right_linear_vel) / self.wheel_base
                
                time_secs = msg.header.stamp.secs
                time_nsecs = msg.header.stamp.nsecs

                self.current_time = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs).to_sec()
                self.dt = self.current_time - self.last_time        
                self.last_time = self.current_time

                uk = np.array([self.v*self.dt, 0, self.w*self.dt]).reshape(3, 1)
                
                self.prediction( self.dt)
           

                self.publish_odometry(msg)
        self.mutex.release()

    """
    Publishes the odometry message
    """
    def publish_odometry(self ,msg):
       
        odom = Odometry()
        
        current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
     
        theta = self.xk[-1].copy()
        q = quaternion_from_euler(0, 0, float(theta))
        
        covar = [self.Pk[-3,-3], self.Pk[-3,-2], 0.0, 0.0, 0.0, self.Pk[-3,-1],
                self.Pk[-2,-3], self.Pk[-2,-2], 0.0, 0.0, 0.0, self.Pk[-2,-1],  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                self.Pk[-1,-3], self.Pk[-1,-2], 0.0, 0.0, 0.0, self.Pk[-1,-1]]

        odom.header.stamp = current_time
        odom.header.frame_id = self.parent_frame
        odom.child_frame_id = self.child_frame
    
        odom.pose.pose.position.x = self.xk[-3]
        odom.pose.pose.position.y = self.xk[-2]

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w
        odom.pose.covariance = covar

        self.odom_pub.publish(odom)
     
        self.tf_broadcaster.sendTransform((self.xk[-3], self.xk[-2], 0.0), q , rospy.Time.now(), self.child_frame, self.parent_frame)
   
    def imu_callback(self, msg):
        self.mutex.acquire()
        orientation = msg.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]

        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        self.yaw = np.array([yaw]).reshape(1, 1)
        self.heading_updae = True
        self.heading_update()
        self.mutex.release()
    
    def prediction(self, dt):
    
        # Calculate Jacobians with respect to state vector
        Ak = np.array([[1, 0, -math.sin(float(self.xk[2]))*(self.v)*dt],
                      [0, 1, math.cos(float(self.xk[2]))*(self.v)*dt],
                      [0, 0, 1]])

        # Calculate Jacobians with respect to noise
        Wk = np.array([[0.5 * math.cos(float(self.xk[2]))*dt, 0.5 * math.cos(float(self.xk[2]))*dt],
                       [0.5 * np.sin(float(self.xk[2]))*dt, 0.5 *
                        math.sin(float(self.xk[2]))*dt],
                       [-dt/self.wheel_base, dt/self.wheel_base]])

        # Update the prediction "uncertainty"
        self.Pk = Ak @ self.Pk @ Ak.T + Wk @ self.Qk @ Wk.T

        # Integrate position
        self.xk[0] = self.xk[0] + self.v*math.cos(float(self.xk[2]))*dt
        self.xk[1] = self.xk[1] + self.v*math.sin(float(self.xk[2]))*dt
        self.xk[2] = self.wrap_angle(self.xk[2] + (self.w)*dt)
       


#############################
        # Update step
############################
    def heading_update(self):
        # Create a row vector of zeros of size 1 x 3*num_poses
        self.compass_Vk = np.diag([0.001])
        # define the covariance matrix of the compass
        self.compass_Rk = np.diag([0.01]) 
         
        Hk = np.zeros((1, len(self.xk)))
        # Replace the last element of the row vector with 1
        Hk[0, -1] = 1
        predicted_compass_meas = self.xk[-1]
        # Compute the kalman gain
        K = self.Pk @ Hk.T @ np.linalg.inv((Hk @ self.Pk @ Hk.T) + (self.compass_Vk @ self.compass_Rk @ self.compass_Vk.T))

        # Compute the innovation
        innovation = np.array(self.wrap_angle(self.yaw[0] - predicted_compass_meas)).reshape(1, 1)

        # Update the state vector
        
        self.xk = self.xk + K@innovation

        # Create the identity matrix        
        I = np.eye(len(self.xk))

        # Update the covariance matrix
        self.Pk = (I - K @ Hk) @ self.Pk @ (I - K @ Hk).T

    def velocity_callback(self, msg):
        
        lin_vel = msg.linear.x
        ang_vel = msg.angular.z

        # print("linear and angular ", lin_vel , ang_vel )
        left_linear_vel   = lin_vel  - (ang_vel*self.wheel_base/2)
        right_linear_vel = lin_vel  +  (ang_vel*self.wheel_base/2)
 
        left_wheel_velocity  = left_linear_vel / self.wheel_radius
        right_wheel_velocity = right_linear_vel / self.wheel_radius
            
        wheel_vel = Float64MultiArray()
        wheel_vel.data = [left_wheel_velocity, right_wheel_velocity]
        self.vel_pub.publish(wheel_vel)



# Main function
if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('Odometry_node')

    # Create an instance of the DifferentialDrive class
    diff_drive = Odom()
    # Spin
    rospy.spin()