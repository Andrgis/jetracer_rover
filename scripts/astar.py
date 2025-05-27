#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS Melodic node wrapping the A* planner.
Publishes two topics:
  - /planned_path (nav_msgs/Path) for debugging
  - /cmd_vel_actions (geometry_msgs/Twist) sequence of actions to reach goal
Subscribes to:
  - /move_base_simple/goal (geometry_msgs/PoseStamped)
  - /amcl_pose (geometry_msgs/PoseWithCovarianceStamped) for current rover pose
"""
import rospy
import math
import heapq
import os
import yaml
import numpy as np
from scipy.ndimage import distance_transform_edt
import tf
from tf import transformations
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, Quaternion, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Header

# Utility functions
def read_yaml(filename):
    with open(filename) as f:
        return yaml.safe_load(f)

def read_pgm(pgmf):
    magic = pgmf.readline()
    if magic != b'P5\n':
        raise ValueError("Not a binary PGM (P5) file")
    line = pgmf.readline().decode('ascii')
    while line.startswith('#'):
        line = pgmf.readline().decode('ascii')
    width, height = map(int, line.split())
    depth_line = pgmf.readline().decode('ascii')
    while depth_line.startswith('#'):
        depth_line = pgmf.readline().decode('ascii')
    maxval = int(depth_line)
    data = np.frombuffer(pgmf.read(), dtype=np.uint8)
    return data.reshape((height, width))

def threshold_map(raster):
    # treat values >250 as free (0), else occupied (1)
    return (raster <= 250).astype(np.uint8)

def world_to_map(wx, wy, origin_x, origin_y, resolution, height):
    px = (wx - origin_x) / resolution
    py = (wy - origin_y) / resolution
    ix = int(px)
    iy = int(height - 1 - py)
    return ix, iy

def T_from_pose(x,y,th):
    T = np.array([[math.cos(th), -math.sin(th), x],
                  [math.sin(th), math.cos(th), y],
                  [0.0, 0.0, 1.0]])
    return T

def pose_from_T(T):
    x = T[0, 2]
    y = T[1, 2]
    th = math.atan2(T[1, 0], T[0, 0])
    return x, y, th

# Node class for A*
class Node:
    def __init__(self, x, y, theta, cost, parent, action):
        self.x = x; self.y = y; self.theta = theta; self.cost = cost; self.parent = parent; self.action = action
    def __lt__(self, other): return self.cost < other.cost  # Defining comparision

# Heuristic (Euclidean + angle)
def heuristic(node, goal, h_weight=1.1):
    # Simplified Reeds-Shepp for common cases
    dx = goal.x - node.x
    dy = goal.y - node.y
    dtheta = (goal.theta - node.theta + math.pi) % (2 * math.pi) - math.pi

    dist = math.hypot(dx, dy)

    # If we're already pointing roughly toward goal
    target_angle = math.atan2(dy, dx)
    angle_to_target = (target_angle - node.theta + math.pi) % (2 * math.pi) - math.pi

    # Estimate minimum path considering turns
    turn_radius = 10.0  # Adjust based on your robot's turning radius

    if abs(angle_to_target) < 0.1:  # Nearly aligned
        return dist + abs(dtheta) * turn_radius * h_weight
    else:
        # Need to turn, move, then align
        turn_cost = abs(angle_to_target) * turn_radius
        final_turn_cost = abs(dtheta) * turn_radius
        return turn_cost + dist + final_turn_cost * h_weight

def proximity_penalty(dist_map, x, y):
    clearance = 10.0; w = 20.0
    d = dist_map[int(y), int(x)]
    return 0.0 if d >= clearance else w * (1.0 - d/clearance) ** 2

def is_collision(x, y, grid):
    ix, iy = int(x), int(y)
    if ix < 0 or iy < 0 or iy >= grid.shape[0] or ix >= grid.shape[1]:
        return True
    return grid[iy, ix] == 1

# A* planner
def a_star(start, goal, grid, dist_map, res):
    # action definitions
    rospy.loginfo('Running A* path planning')
    ang_vel = 0.6
    turn_radius = 0.5/res
    swing_vel = turn_radius*ang_vel
    vel = 0.5 / res
    cost_scale = 1.0; back_penalty = 1.5; swing_penalty = 1.3
    actions = [
        (vel, 0.0, vel * cost_scale), # forward
        (swing_vel, ang_vel, swing_vel * swing_penalty * cost_scale), # Left forward
        (swing_vel, -ang_vel, swing_vel * swing_penalty * cost_scale), # Right forward
        (-vel, 0.0, back_penalty * vel * cost_scale), # backward
        (-swing_vel, ang_vel, back_penalty * swing_vel * swing_penalty * cost_scale), # Right backward
        (-swing_vel, -ang_vel, back_penalty * swing_vel * swing_penalty * cost_scale) # Left backward
    ]
    tol_px = 0.25 / res
    tol_angle = 0.6
    open_list = []
    heapq.heappush(open_list, (0, start))
    closed = set()
    max_iter = 200000
    i = 0
    initial_dist = math.hypot(start.x - goal.x, start.y - goal.y)
    while open_list and i < max_iter:
        _, cur = heapq.heappop(open_list)
        key = (cur.x//2, cur.y//2, (cur.theta*12)//(2*math.pi)%12)
        if key in closed:
            continue
        closed.add(key)

        # Add progress tracking
        i += 1
        if i % 1000 == 0:
            best_dist = math.hypot(cur.x - goal.x, cur.y - goal.y)
            if i > 50000 and best_dist > initial_dist * 0.8:  # Not making progress
                rospy.loginfo("Search stalled, increasing tolerances by 50 %")
                tol_px *= 1.5
                tol_angle *= 1.5
        # check if goal reached
        angle_diff = abs((cur.theta - goal.theta + math.pi) % (2 * math.pi) - math.pi)
        if abs(cur.x - goal.x) < tol_px and abs(cur.y - goal.y) < tol_px and angle_diff < tol_angle:
            print('Found path after %d iterations', i)
            path = []
            action_path = []
            while cur:
                path.append((cur.x, cur.y, cur.theta))
                if cur.action:
                    action_path.append(cur.action)
                cur = cur.parent
            return path[::-1], action_path[::-1]
        # expand
        for v, w, c in actions:
            new_theta = cur.theta + w
            new_theta = (new_theta + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-π, π]
            if w == 0:
                dx = v * math.cos(cur.theta)
                dy = v * math.sin(cur.theta)
            else:
                dx = v / w * (math.sin(new_theta) - math.sin(cur.theta))
                dy = -v / w * (math.cos(new_theta) - math.cos(cur.theta))
            nx, ny = cur.x + dx, cur.y + dy
            if not is_collision(nx, ny, grid):
                pen = proximity_penalty(dist_map, nx, ny)
                nc = cur.cost + c + pen
                node = Node(nx, ny, new_theta, nc, cur, (v, w))
                f = nc + heuristic(node, goal)
                heapq.heappush(open_list, (f, node))
    return []

# ROS node
class AStarPlannerNode(object):
    def __init__(self):
        rospy.init_node('astar_planner')
        # load map
        map_yaml = rospy.get_param('~map_yaml')
        map_dir = os.path.dirname(map_yaml)
        m = read_yaml(map_yaml)
        yaml_img = m['image']
        pgm_path = os.path.join(map_dir, yaml_img)
        self.res = m['resolution']
        with open(pgm_path, 'rb') as f:
            raster = read_pgm(f)
        self.height, self.width = raster.shape
        self.grid = threshold_map(raster)
        self.dist_map = distance_transform_edt(self.grid == 0)
        self.origin_x, self.origin_y, self.origin_th = m['origin']
        self.T_wo = np.array([[1.0,0.0,-self.origin_x/self.res], [0.0,-1.0,-self.origin_y/self.res],[0,0,1]])
        self.T_oi = np.array([[1.0,0.0,0.0], [0.0,1.0,0.0],[0.0,0.0,1.0]])
        self.T_ir = np.array([[1.0,0.0,0.0], [0.0,1.0,0.0],[0.0,0.0,1.0]])
        # publishers and subscribers
        self.path_pub = rospy.Publisher('planned_path', Path, queue_size=1)
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('move_base_simple/goal', PoseStamped, self.goal_cb)
        rospy.loginfo('A* planner ready')
        rospy.spin()

    def goal_cb(self, msg):
        # get start pose in world coords
        amcl = rospy.wait_for_message('amcl_pose', PoseWithCovarianceStamped)
        wx_s = amcl.pose.pose.position.x
        wy_s = amcl.pose.pose.position.y
        _, _, th_s = tf.transformations.euler_from_quaternion([
            amcl.pose.pose.orientation.x,
            amcl.pose.pose.orientation.y,
            amcl.pose.pose.orientation.z,
            amcl.pose.pose.orientation.w])
        self.T_oi = T_from_pose(wx_s/self.res,wy_s/self.res,th_s)
        wx_g = msg.pose.position.x
        wy_g = msg.pose.position.y
        _, _, th_g = tf.transformations.euler_from_quaternion([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w])
        odom = rospy.wait_for_message('odom_combined', PoseWithCovarianceStamped)
        wx_odom = odom.pose.pose.position.x
        wy_odom = odom.pose.pose.position.y
        _, _, th_odom = tf.transformations.euler_from_quaternion([
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w])
        self.T_ir = T_from_pose(wx_odom/self.res, wy_odom/self.res, th_odom)
        # convert to map indices
        ix_s, iy_s = world_to_map(wx_s, wy_s, self.origin_x, self.origin_y, self.res, self.height)
        ix_g, iy_g = world_to_map(wx_g, wy_g, self.origin_x, self.origin_y, self.res, self.height)
        start = Node(ix_s, iy_s, th_s, 0, None, None)
        goal = Node(ix_g, iy_g, th_g, 0, None, None)
        curr = Node(ix_s, iy_s, th_s, 0, None, None)
        rospy.loginfo("[AMCL & msg] start_raw=(%d, %d, %.2f) goal_raw=(%d, %d, %.2f)", wx_s, wy_s, th_s, wx_g, wy_g, th_g)
        rospy.loginfo("[Odom] curr = (%d, %d, %.2f)", wx_odom, wy_odom, th_odom)
        rospy.loginfo("[A*] start=(%d, %d, %.2f) goal=(%d, %d, %.2f)", ix_s, iy_s, th_s, ix_g, iy_g, th_g)
        # MPC time
        path_actions_mpc = []
        horizon = 4
        while abs(curr.x-goal.x)>5 or abs(curr.y-goal.y)>5 or abs((curr.theta-goal.theta+math.pi)%(2*math.pi)-math.pi)>0.5:
            _, path_actions = a_star(curr, goal, self.grid, self.dist_map, self.res)
            path_actions_mpc += path_actions[:horizon]
            rospy.loginfo("[MPCA*] actions planned: %s", path_actions[:horizon])
            for a in path_actions[:horizon]:
                twist = Twist()
                twist.linear.x = a[0] * self.res
                twist.angular.z = -a[1]
                self.cmd_pub.publish(twist)
                rospy.sleep(2.0)
            # Update current position from odometry
            odom = rospy.wait_for_message('odom_combined', PoseWithCovarianceStamped)
            wx_c = odom.pose.pose.position.x
            wy_c = odom.pose.pose.position.y
            _, _, th_c = tf.transformations.euler_from_quaternion([
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w])
            self.T_ir = T_from_pose(wx_c/self.res, wy_c/self.res, th_c)
            rospy.loginfo("[Odom] Current=(%d, %d, %.2f)", wx_c, wy_c, th_c)
            ix, iy, th_c = pose_from_T(self.T_wo*self.T_oi*self.T_ir)
            rospy.loginfo("[MPCA*] Current=(%d, %d, %.2f)", ix, iy, th_c)
            curr = Node(ix, iy, th_c, 0, None, None)
        rospy.loginfo("[MPCA*] Done.")

        #if not path_states:
        #    rospy.logwarn("[A*] no path found!")
        #    return
        ## publish planned_path in world coords
        #ros_path = Path(); ros_path.header = Header(frame_id='map')
        #for n in path_states:
        #    ps = PoseStamped(); ps.header = ros_path.header
        #    wx_c = self.origin_x+n.x*self.res
        #    wy_c = self.origin_y+(self.height-1-n.y)*self.res
        #    ps.pose.position = Point(wx_c, wy_c, 0)
        #    quat = tf.transformations.quaternion_from_euler(0, 0, n.theta)
        #    ps.pose.orientation = Quaternion(*quat)
        #    ros_path.poses.append(ps)
        #self.path_pub.publish(ros_path)
        #for n in path_states:
        #    ps = PoseStamped(); ps.header = ros_path.header
        #    wx = self.origin_x + n.y * self.res
        #    wy = self.origin_y + (self.height - 1 - n.x) * self.res
        #    ps.pose.position = Point(wx, wy, 0)
        #    quat = tf.transformations.quaternion_from_euler(0, 0, n.theta)
        #    ps.pose.orientation = Quaternion(*quat)
        #    ros_path.poses.append(ps)
        #self.path_pub.publish(ros_path)
        # publish cmd_vel_actions


if __name__ == '__main__':
    try:
        AStarPlannerNode()
    except rospy.ROSInterruptException:
        pass
