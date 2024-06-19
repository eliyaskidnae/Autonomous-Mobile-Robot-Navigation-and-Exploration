#!/usr/bin/env python3
import py_trees
import rospy
from std_srvs.srv import Trigger, TriggerRequest
from std_msgs.msg import String, Float32MultiArray  
from pick_up_objects_task.srv import Waypoint, WaypointRequest
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from time import sleep
from py_trees.composites import Sequence , Parallel , Selector
from py_trees import logging as log_tree 
from py_trees.decorators import Inverter , Retry , Timeout
from geometry_msgs.msg import PoseStamped
from project_handsonintervention.msg import CustomPoseStamped 
from nav_msgs.msg import Odometry
from std_msgs.msg import String 
from std_srvs.srv import SetBool
from std_msgs.msg import Bool
import tf
import numpy as np


# Behiviour to check if exploration is done or not
class Check_Exploration_Done(Behaviour):
  def __init__(self, name):
    super(Check_Exploration_Done, self).__init__(name)

    self.blackboard = self.attach_blackboard_client(name=self.name)
    self.blackboard.register_key( "exploration_done", access= py_trees.common.Access.READ)
    self.blackboard.register_key( "exploration_done", access= py_trees.common.Access.WRITE)

  def setup(self):
    self.logger.debug(f"Check_Exploration_Done::setup {self.name}")
    self.exploration_done = False
    self.blackboard.exploration_done = self.exploration_done

  def initialise(self):
    self.logger.debug(f"Check_Exploration_Done::initialise {self.name}")
    self.exploration_done = self.blackboard.exploration_done

  def update(self):
    self.logger.debug(f"Check_Exploration_Done::update {self.name}")
    if(self.exploration_done):
      print("Explore Done" , self.exploration_done)
      return Status.SUCCESS
    # check if miniman wavepoint is visited or not
    return Status.FAILURE

  def terminate(self, new_status):
    self.logger.debug(f"Check_Exploration_Done::terminate {self.name} to {new_status}")
    
class Explore(Behaviour):
  def __init__(self, name):
    super(Explore, self).__init__(name)

    self.blackboard = self.attach_blackboard_client(name=self.name)
    self.blackboard.register_key( "locations", access= py_trees.common.Access.WRITE) # list of waypoints
    self.blackboard.register_key( "next_waypoint", access= py_trees.common.Access.WRITE) # goal point 
    self.blackboard.register_key( "exploration_done", access= py_trees.common.Access.WRITE)
    self.get_frontier = rospy.Subscriber('/frontier_goal', PoseStamped, self.frontier_callback)
    self.frointer = False

  def setup(self):
    self.logger.debug(f"Explore::setup {self.name}")
    
    self.move_goal_pub = rospy.Publisher(
            '/move_base_simple/goal', PoseStamped, queue_size=10)
    self.logger.debug(
                "  %s [Explore::setup() Server connected!]" % self.name)

  def frontier_callback(self, msg):
    print("Frontier Goal", msg.pose.position.x, msg.pose.position.y)
    self.frointer = True
    print("Frontier 1", self.frointer)
    
    self.frontier = True

    self.locations.append([msg.pose.position.x, msg.pose.position.y])
    self.blackboard.locations = self.locations

  def initialise(self):
    self.logger.debug(f"Explore::initialise {self.name}")
    self.locations = []
  def update(self):
    self.logger.debug(f"Explore::update {self.name}")
    if self.locations and int(self.locations[0][0]) == 1000: 
        print("EEEExploration Done")
        self.blackboard.exploration_done = True
        return Status.FAILURE
      
    elif self.frointer:  
        goal = PoseStamped()
        self.next_waypoint = self.blackboard.locations[0]
        self.blackboard.next_waypoint = self.next_waypoint
        del self.blackboard.locations[0]
        self.frointer = False
        return Status.SUCCESS
    
    else:
        return Status.RUNNING
      
  def terminate(self, new_status):
    self.logger.debug(f"Explore::terminate {self.name} to {new_status}")
    
class Set_Exploration_Done(Behaviour):
  def __init__(self, name):
    super(Set_Exploration_Done, self).__init__(name)
    
    self.blackboard = self.attach_blackboard_client(name=self.name)
    self.blackboard.register_key( "exploration_done", access= py_trees.common.Access.WRITE)
    
  def setup(self):
    self.logger.debug(f"explaration done::setup {self.name}")
    
  def initialise(self):
    self.logger.debug(f"explaration done::initialise {self.name}")
   
  def update(self):
    self.logger.debug(f"explaration done::update {self.name}")
    self.blackboard.exploration_done = True
    return Status.FAILURE
    

  def terminate(self, new_status):
    self.logger.debug(f"Path_Follower::terminate {self.name} to {new_status}")


class Planner(Behaviour):
  def __init__(self, name):
    super(Planner, self).__init__(name)

    self.blackboard = self.attach_blackboard_client(name=self.name)
   
    self.blackboard.register_key( "next_waypoint", access= py_trees.common.Access.READ) # goal point 

  def setup(self):
    self.logger.debug(f"Explore::setup {self.name}")
    
    self.move_goal_pub = rospy.Publisher(
            '/move_base_simple/goal', PoseStamped, queue_size=10)
    self.logger.debug(
                "  %s [Explore::setup() Server connected!]" % self.name)


  def initialise(self):
    self.logger.debug(f"Explore::initialise {self.name}")
    self.waypoint = self.blackboard.next_waypoint
    
  def update(self):
    self.logger.debug(f"Explore::update {self.name}")

    if self.waypoint:  
        goal = PoseStamped()
        goal.pose.position.x = self.waypoint[0]
        goal.pose.position.y = self.waypoint[1]
        self.move_goal_pub.publish(goal)
        # print("FrontierGoal Point", self.waypoint)
        self.frointer = False
        return Status.SUCCESS
    else:
        # print("No Frontier Goal Point")
        return Status.FAILURE
  def terminate(self, new_status):
    self.logger.debug(f"Explore::terminate {self.name} to {new_status}")
    
# Behavior to follow the path
class Path_Follower(Behaviour):
  def __init__(self, name):
    super(Path_Follower, self).__init__(name)
    
    self.blackboard = self.attach_blackboard_client(name=self.name)

    self.blackboard.register_key( "next_waypoint", access= py_trees.common.Access.WRITE)
    self.blackboard.register_key( "next_waypoint", access= py_trees.common.Access.READ)
    rospy.Subscriber('/goal_reached', Bool, self.get_goal_status)
    self.goal_reached = False
    rospy.Subscriber('/odom', Odometry, self.odom_callback)
    self.distance_threshold = 0.15
  def odom_callback(self, data):
        self.robot_pose = (data.pose.pose.position.x,
                           data.pose.pose.position.y)
  def get_goal_status(self, msg):
    print("Goal Reached", msg.data)
    if msg.data:
       
        self.goal_reached = True
  def distance(self, p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
  
  def setup(self):
    self.logger.debug(f"Path_Follower::setup {self.name}")
    
  def initialise(self):
    self.logger.debug(f"Path_Follower::initialise {self.name}")
    self.next_waypoint = self.blackboard.next_waypoint 
  def update(self):
    self.logger.debug(f"Path_Follower::update {self.name}")
    
    try:
        # success if goal point is reached
        print("Goal Point", self.next_waypoint)
        if self.goal_reached:
            self.nex_waypoint = None
            print("Goal Point Reached") 
            return Status.SUCCESS
        # running while following the path
        else:   
            return Status.RUNNING
    except:
        self.logger.debug(
            "  {}: Error Following path".format(self.name))
        return py_trees.common.Status.FAILURE

  def terminate(self, new_status):
    self.logger.debug(f"Path_Follower::terminate {self.name} to {new_status}")
  
if __name__ == "__main__":
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    rospy.init_node("behavior_trees")
    
    explore_list  =   Explore("explore_list")
    # get_frontier =  Timeout( name="get_frontiers", duration=60 ,child=explore_list)
    set_exploration = Set_Exploration_Done("set_exploration")
    explore_finish = Check_Exploration_Done("explore_finish")
    check_exploration = Inverter( "Invertor" , explore_finish )
    path_planner = Planner("plan path")
    path_follower = Path_Follower("plan and follow path")

    # Create a root for the tree
    root = Sequence(name="Frontier Exploration", memory=True)
    log_tree.level = log_tree.Level.DEBUG
    root.add_children(
        [
            check_exploration,
            Selector("Explore_Selec", memory=True, children=[explore_list , set_exploration] ),
            path_planner,
            path_follower
        ])
    root.setup_with_descendants() 
    py_trees.display.render_dot_tree(root)

    try:
        while not rospy.is_shutdown() and check_exploration.status != Status.FAILURE:
            root.tick_once()
            sleep(3) 
    except KeyboardInterrupt:
        print("Shutting down")
        rospy.signal_shutdown("KeyboardInterrupt")
        raise