#!/usr/bin/env python3

import numpy as np
import math
import random
from  utils_lib.rrt_start_debiuns import RRT as RRT_DB
import numpy as np
from time import time
import scipy.spatial
import matplotlib.pyplot as plt
import numpy as np

local_map = None
local_origin = None
local_resolution = None

def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

class StateValidityChecker:
    """ Checks if a position or a path is valid given an occupancy map."""
    # Constructor
    def __init__(self, distance=0.2, is_unknown_valid=True , is_rrt_star = True):
        self.map = None 
        self.resolution = None
        self.origin = None
        self.there_is_map = False
        self.distance = 0.21              
        self.is_unknown_valid = is_unknown_valid  
        self.is_rrt_star = is_rrt_star  
    
    # Set occupancy map, its resolution and origin. 
    def set(self, data, resolution, origin):
        global local_map, local_origin, local_resolution
        self.map = data
        self.resolution = resolution
        self.origin = np.array(origin)
        self.there_is_map = True
        self.height = data.shape[0]
        self.width = data.shape[1]
        local_map = data
        local_origin = origin
        local_resolution = resolution
                
    def get_distance( self , first_pose , second_pose):
        return math.sqrt((second_pose[0] - first_pose[0] )**2 + (second_pose[1] - first_pose[1]) **2)
    
    def is_valid(self, pose):        
        m = self.__position_to_map__(pose)
        grid_distance = int(self.distance/self.resolution)
        # add 6 points upper and to lower limit 
        lower_x , lower_y = m[0] - grid_distance , m[1] - grid_distance  
        for lx in range(0,2*grid_distance):
            for ly in range(0,2*grid_distance):
                pose = lower_x + lx, lower_y + ly              
                # if one of the position is not free return False  , stop the loop 
                if(self.is_onmap(pose)):                           
                    if(not self.is_free(pose)): 
                        return False     
                # if  position is not in the map bounds return  is_unknown_valid
                else:
                    if(not self.is_unknown_valid):
                        return False
        return True
    def min_dis_obstacle(self, p, distance=1):
        
        m = self.__position_to_map__(p)
        grid_distance = int(distance/self.resolution)
        lower_x , lower_y = m[0] - grid_distance, m[1] - grid_distance
        min_dis = float('inf')
        for lx in range(2*grid_distance):
            for ly in range(2*grid_distance):
                pose = lower_x + lx, lower_y + ly
                
                if(not self.is_onmap(pose) or not self.is_free(pose)):
                    distance =  np.linalg.norm(np.array(pose) - np.array(m)) 
                    min_dis = min(min_dis, distance)
        return   min_dis
    
    def not_valid_pose(self, pose): 
        # returns pose that are not valid
        m = self.__position_to_map__(pose)
        grid_distance = int(self.distance/self.resolution)
        # add 6 points upper and to lower limit 
        lower_x , lower_y = m[0] - grid_distance , m[1] - grid_distance  
        for lx in range(0,2*grid_distance):
            for ly in range(0,2*grid_distance):
                pose = lower_x + lx, lower_y + ly    
                # if(self.map[pose[0],pose[1]] >50):
                #     return pose
                # if one of the position is not free return False  , stop the loop 
                if(self.is_onmap(pose)):                           
                    if(not self.is_free(pose)): 
                        return pose     
                # if  position is not in the map bounds return  is_unknown_valid
                else:
                    if(not self.is_unknown_valid):
                        return pose
        return None
    
    def check_path_smooth(self,paths):
        for path in paths:
            if(not self.is_valid(path)):
                return False
        return True
    
    def check_path(self, path):
        step_size = 0.2*self.distance
        valid = True
        for index in range(len(path)-1):
            first_pose  = path[index] # initial path 
            second_pose = path[index+1] # next path 
            # direction vector from first_pose to second_pose
            dir_vector = np.array([second_pose[0] - first_pose[0] , second_pose[1] - first_pose[1]])
            distance = self.get_distance(first_pose , second_pose) # distance from first_pose to second_pose
            
            if distance == 0:
                norm_vect = np.array([0, 0])
            else:
                norm_vect = dir_vector / distance # Compute normal vector b/n two points
            discrtized_seg = np.array([first_pose]) # array of discritized segment 
            current = first_pose

            # while the distance b/n two poses is smaller than min distance 
            valid = self.is_valid(first_pose)
            while(self.get_distance(second_pose , current) > step_size):          
                current  = current + norm_vect*step_size
                valid = self.is_valid(current)      
                # if one of the discritized poses are not vlid return false    
                if(not valid):
                    return False      
            # Finnally Check the goal point
            valid = self.is_valid(second_pose)
        # If each path is valid return true
        return valid
    # Transform position with respect the map origin to cell coordinates
    def __position_to_map__(self, p):      
        x , y  =  p # get x and y positions 
        m_x    =  (x - self.origin[0])/self.resolution  # x cell cordinate 
        m_y    =  (y - self.origin[1])/self.resolution  # y cell cordinate 
        return [round(m_x), round(m_y)]   
    def __map_to_position__(self, m):
            x ,y = m  
            p_x  = self.origin[0]+ x * self.resolution 
            p_y  = self.origin[1] + y * self.resolution
            return [p_x, p_y]  
    def is_onmap(self,pose):    
        if( 0<=pose[0]< self.height and 0<= pose[1] < self.width):
            return True        
        else:
            return False
        # checks a given pose in grid is free  
    def is_free(self, pose): 
        # if is is free return True , which means  
        if self.map[pose[0],pose[1]] == 0 :
            return True, 
        #if it is unkown return opposite of is unkown valid 
        elif self.map[pose[0],pose [1]] == -1 : # return opposite of is unkown valid 
            return  self.is_unknown_valid
        return False   

def compute_path(start_p, goal_p, svc, bounds , max_time=7.0):
    rrt = RRT_DB(svc , 3000 ,0.6, 0.2 , bounds, max_time )
    # returns the smooth path and the tree list
    path  , tree_list = rrt.compute_path(start_p, goal_p )
    # path = rrt.compute_path( start_p , goal_p)
    return path , tree_list
    
def move_to_point(current, goal, Kv=0.5, Kw=0.5):
    d = ((goal[0] - current[0])**2 + (goal[1] - current[1])**2)**0.5
    psi_d = np.arctan2(goal[1] - current[1], goal[0] - current[0])
    psi = wrap_angle(psi_d - current[2])

    v = 0.0 if abs(psi) > 0.05 else Kv * d
    w = Kw * psi
    return v, w

# Controller
def pure_p_control(current, goal):
    # Parameters
    robot_x, robot_y, robot_yaw = current
    if len(goal) == 3:
        goal_x, goal_y, _ = goal
    else:
        goal_x, goal_y = goal

    lookahead_distance = 0.05 # Distance to look ahead on the path
    L = 0.235  # Wheelbase of the robot (distance between front and rear wheels)
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    angle = wrap_angle(math.atan2(dy, dx) - current[2])
    distance_to_goal = math.sqrt(dx**2 + dy**2)
    if distance_to_goal < lookahead_distance: 
        lookahead_x, lookahead_y = goal_x, goal_y
    else: 
        scale = lookahead_distance/distance_to_goal
        lookahead_x = robot_x + scale * dx
        lookahead_y = robot_y + scale * dy
        
    angle_to_lookahead = math.atan2(lookahead_y - robot_y, lookahead_x - robot_x)
    alpha = wrap_angle(angle_to_lookahead - robot_yaw)
    delta = math.atan2(2* L * math.sin(alpha), lookahead_distance)

    v = 0.4
    w = (v / L) * math.tan(delta)
    # if abs(angle) > 0.8 and distance_to_goal < 0.35:
    #     v = 0
    return v, w

# Controller 
def Adaptive_pp_control(current, goal):
    # Parameters
    robot_x, robot_y, robot_yaw = current
    if len(goal) == 3:
        goal_x, goal_y, _ = goal
    else:
        goal_x, goal_y = goal

    lookahead_distance = 0.1  # Distance to look ahead on the path
    L = 0.235  # Wheelbase of the robot (distance between front and rear wheels)
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    angle = wrap_angle(math.atan2(dy, dx) - current[2])
    distance_to_goal = math.sqrt(dx**2 + dy**2)
    if distance_to_goal < lookahead_distance: 
        lookahead_x, lookahead_y = goal_x, goal_y
    else: 
        scale = lookahead_distance/distance_to_goal
        lookahead_x = robot_x + scale * dx
        lookahead_y = robot_y + scale * dy

    angle_to_lookahead = math.atan2(lookahead_y - robot_y, lookahead_x - robot_x)

    alpha = wrap_angle(angle_to_lookahead - robot_yaw)
    
    delta = math.atan2(2* L * math.sin(alpha), lookahead_distance)
    
    max_angle = 0.15 #threshold angle for accelration
    max_v = 0.7
    min_v = 0.5

    if abs(alpha) <= max_angle:
        v = max(min_v, min(max_v, min_v + (max_v - min_v) * (max_angle - abs(alpha)) / max_angle))
    else:
        v = min_v
    # Angular velocity
    w = (v / L) * math.tan(delta)

    if abs(angle) > 0.8 and distance_to_goal < 0.35:
        v = 0
    return v, w


