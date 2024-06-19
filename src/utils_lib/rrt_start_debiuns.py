#! /usr/bin/env python3

# RRT* with Dubins path
import numpy as np 
from matplotlib import pyplot as plt 
from PIL import Image 
from math import sqrt
import time
import random
import dubins
import math
from scipy.spatial import cKDTree
# class that represent a node in the RRT tree
class Node:
    def __init__(self , x , y  ,yaw = 0):
        self.x = x 
        self.y = y 
        self.yaw = yaw 
        self.id = 0 # 
        self.f_score = float('inf')
        self.g_score = float('inf') 
        self.parent  = None
    # calculate the huristic value of the node
    def calcu_huristic(self,target):
        distance = sqrt( (self.x-target.x)**2 + (self.y-target.y)**2 )
        return distance 
    # calculate the distance between the current node and the target node
    def get_distance(self,target):
        distance = sqrt( (self.x-target.x)**2 + (self.y-target.y)**2 )
        return distance 
    # find the nearest node from the given nodes with in the radius for RRT* method
    def find_nearest_node(self,nodes ):
        min_distance = float('inf')
        nearest_node = None
        for node in nodes:
            distance = self.get_distance(node)
            # if(distance < min_distance and distance != 0 and node != target):
            if(distance < min_distance and distance >  0.0001 and node != self):
                nearest_node = node
                min_distance = distance
            
        return nearest_node
    
    def find_nodes_with_in_radius(self, nodes, radius , k = 20):

        nodes = list(nodes)
        nodes_array = np.array([(node.x, node.y) for node in nodes])
        # Create a KDTree from nodes
        tree = cKDTree(nodes_array)
        # Find the minimum of k and the number of nodes
        k = min(k, len(nodes))
        # Query the tree for k nearest nodes
        distances, indices = tree.query([self.x, self.y], k)
       
        if k == 1:
            distances = np.array([distances])
            indices = np.array([indices])

        distances = distances.flatten()
        indices = indices.flatten()
        # Filter the k nearest nodes based on the radius
        nodes_with_in_radius = [nodes[n] for i,n  in enumerate(indices) if distances[i] <= radius and nodes[n] != self]

        return nodes_with_in_radius
    def __str__(self):
        return str(self.id)
# class that represent the RRT tree
class RRT:
    def __init__(self,svc,k,q,p,dominion = [-10,10,-10,10] ,max_time = 0.1, is_RRT_star = True):
        
        self.svc      = svc
        self.k        = k
        self.q        = q
        self.p        = p 
        # self.dominion = dominion
        self.dominion  = [-10,10,-10,10]
        self.node_list = {}
        self.max       = max_time 
        self.vertices  =    []
        self.edges     =      [ ]
        self.node_counter  =  1
        self.path          = [ ]
        self.smoothed_path = [ ]
        self.goal_index    = []
        self.is_RRT_star   = True # by deafault it is False we implement RRT
        self.radius   = 7    # radius for RRT* search  method
        self.max_time = 7 # max time for the search
        self.goal_found = False
        self.step_size = 0.1
        self.debiun_radius = 0.2

    def cost_optimal_parent(self,qnew,current_parent):
        self.vertices = self.node_list.keys()
        # filter a nodes with a given radius 
        nodes_with_in_radius = qnew.find_nodes_with_in_radius(self.vertices,self.radius)
        best_cost = self.node_list[current_parent] + qnew.get_distance(current_parent)
        # best_cost = current_parent.g_score + qnew.get_distance(current_parent)
        best_parent = current_parent
        path = []
        if(not nodes_with_in_radius): # return the nearest node
            return best_parent , []
        else :  
            for node in nodes_with_in_radius:
                # Get the yaw of the node
                yaw = self.wrap_angle(math.atan2(qnew.y - node.y, qnew.x - node.x))
                
                # collision_free = self.svc.check_path([[node.x,node.y],[qnew.x,qnew.y]])
                collision_free , dubins_path = self.debiuns_check(node,qnew)
                new_node_cost = self.node_list[node] + qnew.get_distance(node)
                new_node_cost = self.node_list[node] +len(dubins_path)
                
                if(new_node_cost < best_cost and collision_free):
                    best_parent = node
                    path = dubins_path
            return best_parent , path

    def rewire(self,qnew):
        # filter a nodes with a given radius 
        self.vertices = self.node_list.keys()
        nodes_with_in_radius = qnew.find_nodes_with_in_radius(self.vertices,self.radius)
        for node in  nodes_with_in_radius:
            # new_node_cost = self.node_list[node] + qnew.get_distance(node)
            new_node_cost = self.node_list[node] + len(qnew.debiuns_path)
            collision_free , debiuns_path = self.debiuns_check(qnew,node)
            # collision_free = self.svc.check_path([[qnew.x,qnew.y],[node.x,node.y]])   
            if(new_node_cost < self.node_list[node] and collision_free):
                node.parent = qnew
                self.edges.remove((node.parent,node))
                self.edges.append((qnew,node))
                self.node_list[node] = new_node_cost
                self.node.debiuns_path = debiuns_path
                self.node.yaw = math.atan2(qnew.y - node.y, qnew.x - node.x)

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.vertices:
            if node.parent == parent_node:
                node.g_score = parent_node.g_score + node.get_distance(parent_node)
                self.propagate_cost_to_leaves(node)

    def Rand_Config(self):       
        
        prob = random.random() # generate a random number between 0 and 1
        # generate a random node with in the dominion 
        x = random.uniform(self.dominion[0],self.dominion[1])
        y = random.uniform(self.dominion[2],self.dominion[3])
        qrand = Node(x,y)
        # if the random number is less than the probability of selecting the goal
        if(prob < self.p):          
           qrand = self.goal # set initialy qrand as goal

        return qrand            
    def Near_Vertices(self , qrand):
        self.vertices = self.node_list.keys()
        qnear =  qrand.find_nearest_node(self.vertices ) # find the nearest node from the vertices
        return qnear
    def New_Config(self , qnear , qrand):
        dir_vector = np.array([qrand.x -  qnear.x , qrand.y - qnear.y])
        length = qnear.get_distance(qrand)
        if(length == 0):
            return qrand
        norm_vector = dir_vector/length

        if(self.q > length):
            return qrand
        qnew = np.array([qnear.x,qnear.y]) + norm_vector*self.q      
        qnew = Node(qnew[0] , qnew[1])
        return qnew
        
    def reconstract_db_path(self):

        self.debiuns_path = []
        self.path = []
        node = self.goal
         
        while node.parent:# 
            for p in (reversed(node.debiuns_path)):
                self.debiuns_path.append(p)
            self.path.append((node.x, node.y))
            node = node.parent


        self.path.append((self.start.x,self.start.y)) #finally add start point 
        self.path.reverse()
        self.debiuns_path.reverse()
  
        return self.debiuns_path
        
    def reconstract_path(self):
        self.vertices = self.node_list.keys()
      
        current = self.goal
        self.path = [current]
        while( current != self.start):
            current = current.parent
            self.path.append(current)
        path =[(n.x,n.y) for n in self.path]
        self.path.reverse()
        path =[[n.x,n.y] for n in self.path]
        return path
    
    def wrap_angle(self,angle):
       return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    def calc_distance_and_angle(self , from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = self.wrap_angle(math.atan2(dy, dx))
        return d, theta 
    def smooth_paths(self): 
        if len(self.path)!=0:
            self.smoothed_path.append(self.path[-1])
            current_pos=self.path[-1]
            current_index=self.path.index(current_pos)
            while (self.path[0] in self.smoothed_path) == False:
                new_list=self.path[0:current_index]
                for i in new_list:
                    if (self.svc.check_path([[i.x , i.y],[current_pos.x  , current_pos.y]])):
                        self.smoothed_path.append(i)
                       
                        current_pos=i
                        current_index=self.path.index(current_pos)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                        break
        self.smoothed_path=list(reversed(self.smoothed_path))
        new_path =[(n.x,n.y) for n in  self.smoothed_path]
        x_path = [n.x for n in self.path]
        angles = []
    
        for i in range(len(self.smoothed_path)-1):
            d , angle = self.calc_distance_and_angle(self.smoothed_path[i],self.smoothed_path[i+1])
            self.smoothed_path[i+1].yaw = angle
            angles.append(angle)
        for i in range(len(self.smoothed_path)-1):
            path_db = dubins.shortest_path((self.smoothed_path[i].x ,self.smoothed_path[i].y , self.smoothed_path[i].yaw), (self.smoothed_path[i+1].x ,self.smoothed_path[i+1].y , self.smoothed_path[i+1].yaw), self.debiun_radius)
            waypoints = path_db.sample_many(self.step_size)
            self.smoothed_path[i+1].waypoints = waypoints

        self.dubin_path = []
        for i in  range(1,len(self.smoothed_path)):
            for (ix , iy ,yaw) in self.smoothed_path[i].waypoints[0]:
                self.dubin_path.append([ix,iy])
        return self.dubin_path
        
    def smooth_path(self):
        counter = 0
        self.smoothed_path = [self.goal]
        while True:
            for node in self.path:
                next_path = self.smoothed_path[len(self.smoothed_path)-1]
                if(self.svc.check_path([[node.x , node.y],[next_path.x, next_path.y]])):
    
                    self.smoothed_path.append(node)
                    break
            if self.smoothed_path[len(self.smoothed_path)-1] == self.start:
                break
            
        
        self.smoothed_path.reverse()
        path =[(n.x,n.y) for n in self.smoothed_path]
        return path
    
    # get the tree of the RRT
    def get_tree(self):
        tree_list = [[[edge[0].x , edge[0].y] ,[edge[1].x , edge[1].y]] for edge in self.edges]
        return tree_list
    def debiuns_check(self,from_node,to_node):
        yaw = self.wrap_angle(math.atan2(to_node.y - from_node.y, to_node.x - from_node.x))
   
        to_yaw = yaw
        check_distance = to_node.get_distance(from_node)
        # check_angel = from_node.yaw - to_yaw
        if False :
        # and check_distance < 0.35:
            path_db = dubins.shortest_path((from_node.x ,from_node.y , to_yaw), (to_node.x ,to_node.y , to_yaw), self.debiun_radius)
        else: 
            path_db = dubins.shortest_path((from_node.x ,from_node.y , from_node.yaw), (to_node.x ,to_node.y , to_yaw), self.debiun_radius)
        waypoints = path_db.sample_many(self.step_size)
      
        debiuns_path = []
        for (ix, iy ,yaw) in (waypoints[0]):
            debiuns_path.append([ix, iy])
        collision_free = self.svc.check_path_smooth(debiuns_path)  
        return collision_free , debiuns_path
    
    def compute_path(self , start , goal):
        self.start_time = time.time()

        self.start = Node(start[0],start[1] ,start[2])   
        self.goal =  Node(goal[0],goal[1])

        self.node_list[self.start] = 0
    
        for k in range(self.k):
            qrand = self.Rand_Config()
            while(not self.svc.is_valid([qrand.x,qrand.y])):
               qrand = self.Rand_Config()
            qnear = self.Near_Vertices(qrand)
            qnew  = self.New_Config(qnear,qrand)
            
            collision_free , debiuns_path = self.debiuns_check(qnear,qnew)
            if(collision_free ):    
                if(True): 
                     qnear,path = self.cost_optimal_parent(qnew,qnear)
                     if(path):
                         debiuns_path = path
                new_cost = self.node_list[qnear] + len(debiuns_path)
               
                if qnew not in self.node_list or self.node_list[qnew] > new_cost:
                    self.node_list[qnew] = new_cost
                    qnew.parent = qnear
                    qnew.debiuns_path = debiuns_path
                    qnew.yaw = math.atan2(qnew.y - qnear.y, qnew.x - qnear.x)
                    self.edges.append((qnear,qnew))   
                    self.node_counter += 1     
          
                if( self.is_RRT_star == True): 
                    self.rewire(qnew)
                    pass
                if(qnew == self.goal):
                    self.goal_found = True
                    # print("goal found")      
            if((time.time() - self.start_time) > self.max_time and not self.goal_found ):
                self.max_time += 0.5 # give additional time to search
            elif(self.goal_found and (time.time() - self.start_time) > self.max_time):
                break 
         
        if(self.goal_found):
            print("max iteration reached")
            return self.reconstract_db_path(), self.path    
        return [], self.get_tree()

    def search_best_goal_node(self):

        goal_indexes = []
        for (i, node) in enumerate(self.vertices):
            if node.get_distance(self.goal) <= self.goal_xy_th:
                goal_indexes.append(i)
        if not goal_indexes:
            return None

        min_cost = min([self.vertices[i].cost for i in goal_indexes])
        for i in goal_indexes:
            if self.vertices[i].g_cost == min_cost:
                return i

        return None