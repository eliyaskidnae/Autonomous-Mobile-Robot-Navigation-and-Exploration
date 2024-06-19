#!/usr/bin/python3

import argparse
import numpy as np
import rospy
import tf
import math
import random
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker , MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA 
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
import time
from utils_lib.online_planning import StateValidityChecker, compute_path , wrap_angle, pure_p_control, Adaptive_pp_control

class OnlinePlanner:
    # OnlinePlanner Constructor
    def __init__(self, gridmap_topic, odom_topic, cmd_vel_topic, bounds, distance_threshold ,is_unknown_valid ,is_rrt_star):
        self.path = []
        self.edges = []
        self.smooth_path = []
        self.trajectory  = []
        self.xk = np.zeros((0,1))
        # State Validity Checker object                                                 
        self.svc = StateValidityChecker(distance_threshold , is_unknown_valid , is_rrt_star)
        # Current robot SE2 pose [x, y, yaw], None if unknown            
        self.current_pose = None
        # Goal where the robot has to move, None if it is not set                                                                   
        self.goal = None
        # Last time a map was received (to avoid map update too often)                                                
        self.last_map_time = rospy.Time.now()
        self.last_odom_time = rospy.Time.now()
        # Dominion [min_x_y, max_x_y] in which the path planner will sample configurations                           
        self.bounds = bounds
        # Tolerance for the distance between the robot and the goal                                        
        self.tolorance = 0.05
        self.Kv = 0.5
        # Proportional angular velocity controller gain                   
        self.Kw = 0.5
        # Maximum linear velocity control action                   
        self.v_max = 0.7
        self.start_time = rospy.Time.now()
        # Maximum angular velocity control action               
        self.w_max = 0.5            
        self.retry = 0 # Retry counter for planning failures
        self.wheel_radius = 0.035 # meters      
        self.wheel_base_distance = 0.235  # meters  
        self.v = 0
        self.w = 0
        self.trial = 0
        self.viewpoints_pub = rospy.Publisher("/slam/vis_viewpoints",MarkerArray,queue_size=1)
        self.cmd_pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray, queue_size=1)
        # Publisher for visualizing the path to with rviz
        self.marker_pub = rospy.Publisher('~path_marker', Marker, queue_size=1)
        self.marker_pub2 = rospy.Publisher('~path_marker_rrt', Marker, queue_size=1)
        self.tree_pub = rospy.Publisher('~tree_marker', Marker, queue_size=1)
        self.goal_reach = rospy.Publisher('/goal_reached', Bool, queue_size=1)
        self.trajetory_pub = rospy.Publisher('~robot_trajectory', Marker, queue_size=1)
        # SUBSCRIBERS
        self.gridmap = rospy.Subscriber(gridmap_topic, OccupancyGrid, self.get_gridmap)
        self.odom = rospy.Subscriber(odom_topic, Odometry, self.get_odom)
        self.flag = True
        # self.move_goal_sub = None # TODO: subscriber to /move_base_simple/goal plublished by rviz 
        self.move_goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.get_goal, queue_size=1)
        self.cmd = Twist()
        # Timer for velocity controller
        rospy.Timer(rospy.Duration(0.01), self.controller)
    
    def get_odom(self, odom):
    
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                              odom.pose.pose.orientation.y,
                                                              odom.pose.pose.orientation.z,
                                                              odom.pose.pose.orientation.w])
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
      
        pose_stamped = PoseStamped()
        pose_stamped.header = odom.header
        pose_stamped.pose = odom.pose.pose
        pose = [pose_stamped.pose.position.x, pose_stamped.pose.position.y]
        self.trajectory.append(pose)
        timer = (odom.header.stamp - self.last_odom_time).to_sec()
    
    def get_goal(self, goal):
        if self.svc.there_is_map: 
            self.trial = 0
            self.goal = np.array([goal.pose.position.x, goal.pose.position.y])  
            if(not self.svc.is_valid(self.goal)):
                rospy.logwarn("Goal Point is not valid , please try again")
                msg = Bool()
                msg = True
                self.goal_reach.publish(msg)
            elif(not self.svc.is_valid(self.current_pose[0:2])):      
                rospy.logwarn("Start Point is not valid , please try again")
                self.recovery_behavior() # move around to find a valid point
            else :
                msg = Bool()
                msg = False
                self.goal_reach.publish(msg)
            # Plan a new path to self.goal
                self.plan()
                
    def rotate_to_explore(self):
        start_time = rospy.Time.now()
        duration = 8
        while(rospy.Time.now() - start_time).to_sec() < duration:
            self.__send_commnd__(0, self.w_max)  
        self.__send_commnd__(0, 0)
        
    def move_back(self):
        start_time = rospy.Time.now()
        duration = 1
        v =0.3
        while(rospy.Time.now() - start_time).to_sec() < duration:
        
            self.__send_commnd__(-v , 0)  
        self.__send_commnd__(0, 0)
        
    def recovery_behavior(self):
        # pass
        pose = self.svc.not_valid_pose(self.current_pose[0:2])
        if pose is not None:
            pose = self.svc.__map_to_position__(pose)
            psi_d = math.atan2(pose[1] - self.current_pose[1] , pose[0]- self.current_pose[0])
    
            angle = math.degrees(wrap_angle(psi_d - self.current_pose[2]))
        
            if( -90 < angle < 90):
                v = -0.2
            else:
                v = 0.2
            start_time = rospy.Time.now()
            duration = 1
            while(rospy.Time.now() - start_time).to_sec() < duration:
                
                self.__send_commnd__(v,0)  
            
            self.__send_commnd__(0, 0)
            self.path = []
            self.trial = 0
            self.plan()
            
    def get_gridmap(self, gridmap):
        if (gridmap.header.stamp - self.last_map_time).to_sec() > 1:            
            self.last_map_time = gridmap.header.stamp
            date = str(rospy.Time.now())
            env = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T
            origin = [gridmap.info.origin.position.x, gridmap.info.origin.position.y]
            self.svc.set(env, gridmap.info.resolution, origin)
      
            if(self.svc.is_valid(self.current_pose[0:2])):
               pass
               self.recovery_behavior() # move around to find a valid point
            if(self.goal is not None and not self.svc.is_valid(self.goal)):
                msg = Bool()
                msg = True 
                self.goal_reach.publish(msg)
            elif self.trial >=5:
                rospy.logwarn("Path Not Found !")
                self.path = []
                self.trial = 0
                self.goal = None
                msg = Bool()
                msg = True 
                self.goal_reach.publish(msg)
            elif self.path is not None and len(self.path) > 0:
                # create total_path adding the current position to the rest of waypoints in the path
                total_path = [self.current_pose[0:2]] + self.path

                if(not self.svc.check_path_smooth(total_path)):
                   
                   rospy.loginfo("Replan agian current  path is in valid ")
                   
                   self.__send_commnd__(0, 0)
                   self.path = []
                   self.plan()
              
    def plan(self):
        self.v = 0
        self.w = 0
        while self.trial < 5:
            # Invalidate previous plan if available
            self.path = []
            if(not self.svc.is_valid(self.goal)):
                # rospy.logwarn("Goal is not valid, Target place is not safe")
                msg = Bool()
                msg = True
                self.goal_reach.publish(msg)

            elif(not self.svc.is_valid(self.current_pose[0:2])):
                rospy.logwarn("Start Point is not valid , please move around ")
                self.recovery_behavior()
            else : 
                self.path , self.tree = compute_path(self.current_pose , self.goal , self.svc , self.bounds)
                if len(self.path) == 0  or []:
                    # rospy.logwarn("Path Not Found !")
                    self.path_not_found = True
                                
                else:
                    self.retry = 0 # reset retry counter
                    self.publish_path()
                    self.publish_path_new(self.tree)
                    # remove initial waypoint in the path (current pose is already reached)
                    if self.path[0]:
                        del self.path[0]                   
                    break   
            self.trial += 1
      
    def controller(self, event):
        if len(self.path)> 0:
            distance_to_goal = self.distance_to_target(self.path[0])
            if (distance_to_goal< self.tolorance):
                del self.path[0]
                if(len(self.path) == 0 ):
                    rospy.loginfo("Goal Point Reached !")
                    self.publish_trajectory(self.trajectory)
                    total_path = self.trajectory_lenght()
                    total_time = (rospy.Time.now() - self.start_time).to_sec()
                    self.xk = np.block([[self.xk],[self.goal.reshape(2,1)]])
                    self.publish_viewpoints()
                   
                    x = self.current_pose[0]
                    y = self.current_pose[1]
                    angle = self.current_pose[2]
                    distance = 0.4
                    # Calculate the next goal position
                    goal_x = x + distance * math.cos(angle)
                    goal_y = y + distance * math.sin(angle)
                    if not self.svc.is_valid((goal_x, goal_y)):
                        self.move_back()
                    self.goal = None
                    msg = Bool()    
                    msg = True
                    self.goal_reach.publish(msg)
                    self.retry = 0
                    self.v = 0
                    self.w = 0
            else:
                self.v, self.w = pure_p_control(self.current_pose, self.path[0])
        self.__send_commnd__(self.v, self.w)
        
    def __send_commnd__(self, v, w):
    
        move = Float64MultiArray() 
        v_l = (2*v + w * self.wheel_base_distance/2) / ( self.wheel_radius)
        v_r = (2*v - w * self.wheel_base_distance/2) / (self.wheel_radius) 
        v_l = v_l/5
        v_r = v_r/5

        move.data = [v_l, v_r]            
        self.cmd_pub.publish(move) 
      
      
    def draw_tree(self):
        if(len(self.edges) > 0):
            m = Marker()
            m.header.frame_id = 'world_net'
            m.header.stamp = rospy.Time.now()
            m.id = 0
            m.type = Marker.LINE_LIST
            m.ns = 'tree'
            m.action = Marker.DELETEALL
            m.lifetime = rospy.Duration(0)
            self.marker_pub.publish(m)

            m.action = Marker.ADD
            m.scale.x = 0.02
            m.scale.y = 0.0
            m.scale.z = 0.0

            m.pose.orientation.x = 0
            m.pose.orientation.y = 0
            m.pose.orientation.z = 0
            m.pose.orientation.w = 1
            color_black = ColorRGBA()
            color_black.r = 0
            color_black.g = 0
            color_black.b = 1
            color_black.a = 1
            for edge in self.edges:
                for node in edge:
                    p = Point()
                    p.x = node[0]
                    p.y = node[1]
                    p.z = 0.0
                    m.points.append(p)
                    m.colors.append(color_black)
            self.tree_pub.publish(m)
            
    def publish_path(self):
        path = self.path.copy()
        if len(self.path) > 1:
            # print("Publish path!")
            m = Marker()
            m.header.frame_id = 'world_ned'
            m.header.stamp = rospy.Time.now()
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.ns = 'path'
            m.action = Marker.DELETE
            m.lifetime = rospy.Duration(0)
            self.marker_pub.publish(m)

            m.action = Marker.ADD
            m.scale.x = 0.04
            m.scale.y = 0.0
            m.scale.z = 0.0
            
            m.pose.orientation.x = 0
            m.pose.orientation.y = 0
            m.pose.orientation.z = 0
            m.pose.orientation.w = 1
            
            color_red = ColorRGBA()
            color_red.r = 0
            color_red.g = 1
            color_red.b = 0
            color_red.a = 1
            color_blue = ColorRGBA()
            color_blue.r = 0
            color_blue.g = 0
            color_blue.b = 1
            color_blue.a = 1

            p = Point()
            p.x = self.current_pose[0]
            p.y = self.current_pose[1]
            p.z = 0.0
            m.points.append(p)
            m.colors.append(color_blue)
            
            for n in path:
                p = Point()
                p.x = n[0]
                p.y = n[1]
                p.z = 0.0
                m.points.append(p)
                m.colors.append(color_red)
            
            self.marker_pub.publish(m)
            
    def publish_path_new(self , path_rrt):
        path = path_rrt
        if len(self.path) > 1:
            m = Marker()
            m.header.frame_id = 'world_ned'
            m.header.stamp = rospy.Time.now()
            m.id = 2
            m.type = Marker.LINE_STRIP
            m.ns = 'path'
            m.action = Marker.DELETE
            m.lifetime = rospy.Duration(0)
            self.marker_pub2.publish(m)

            m.action = Marker.ADD
            m.scale.x = 0.04
            m.scale.y = 0.0
            m.scale.z = 0.0
            
            m.pose.orientation.x = 0
            m.pose.orientation.y = 0
            m.pose.orientation.z = 0
            m.pose.orientation.w = 1
            
            color_red = ColorRGBA()
            color_red.r = 1
            color_red.g = 1
            color_red.b = 0
            color_red.a = 1
            color_blue = ColorRGBA()
            color_blue.r = 0
            color_blue.g = 0
            color_blue.b = 1
            color_blue.a = 1

            p = Point()
            p.x = self.current_pose[0]
            p.y = self.current_pose[1]
            p.z = 0.0
            m.points.append(p)
            m.colors.append(color_blue)
            
            for n in path:
                p = Point()
                p.x = n[0]
                p.y = n[1]
                p.z = 0.0
                m.points.append(p)
                m.colors.append(color_blue)
            self.marker_pub2.publish(m)
            
    def publish_trajectory(self , pose):
        marker = Marker()
        marker.header.frame_id = "world_ned"  
        marker.header.stamp = rospy.Time.now()
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.06 
        marker.color.r = 0.0 
        marker.color.g = 1.0  
        marker.color.a = 1.0  
        for p in self.trajectory:
            point = Point()
            point.x = p[0]
            point.y = p[1]
            marker.points.append(point)
        self.trajetory_pub.publish(marker)
       
    def distance_to_target(self , target):
        return   math.sqrt((self.current_pose[0] - target[0])**2 + (self.current_pose[1] - target[1])**2)
    
    def trajectory_lenght(self):
        lenght = 0
        for i in range(len(self.trajectory)-1):
            lenght += math.sqrt((self.trajectory[i][0] - self.trajectory[i+1][0])**2 + (self.trajectory[i][1] - self.trajectory[i+1][1])**2)
        return lenght
    
    def publish_viewpoints(self):
        # " publish view points"
        marker_frontier_lines = MarkerArray()
        marker_frontier_lines.markers = []

        viewpoints_list = []

        for i in range(0,len(self.xk),2):
            myMarker = Marker()
            myMarker.header.frame_id = "world_ned"
            myMarker.type = myMarker.SPHERE
            myMarker.action = myMarker.ADD
            myMarker.id = i

            myMarker.pose.orientation.x = 0.0
            myMarker.pose.orientation.y = 0.0
            myMarker.pose.orientation.z = 0.0
            myMarker.pose.orientation.w = 1.0

            myPoint = Point()
            myPoint.x = self.xk[i]
            myPoint.y = self.xk[i+1]

            myMarker.pose.position = myPoint
            myMarker.color=ColorRGBA(1, 0, 0, 1)

            myMarker.scale.x = 0.2
            myMarker.scale.y = 0.1
            myMarker.scale.z = 0.05
            viewpoints_list.append(myMarker)

        self.viewpoints_pub.publish(viewpoints_list)
        
# MAIN FUNCTION
if __name__ == '__main__':
    rospy.init_node('turtlebot_online_path_planning_node')  
    is_rrt_star = False
    is_unknown_valid = True
    if rospy.has_param("is_rrt_star"):
        is_rrt_star = bool(rospy.get_param("is_rrt_star")) 
    if rospy.has_param("is_unknown_valid"):
        is_unknown_valid = bool(rospy.get_param("is_unknown_valid"))
    node = OnlinePlanner('/projected_map', '/odom', '/cmd_vel', np.array([-10.0, 10.0, -10.0, 10.0]), 
                         0.2 , is_unknown_valid , is_rrt_star )
    rospy.spin()