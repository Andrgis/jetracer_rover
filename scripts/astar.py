#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS Melodic node wrapping the A* planner.
Fixed coordinate system: +X = East, +Y = North, origin = bottom-left
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
    """Convert world coordinates to map indices with +Y=North convention"""
    px = (wx - origin_x) / resolution  # X remains the same (column index)
    py = (wy - origin_y) / resolution  # Y distance from bottom-left origin
    ix = int(px)
    iy = int(py)  # No longer flip - py directly corresponds to row from bottom
    # Ensure bounds
    iy = max(0, min(height-1, iy))
    ix = max(0, min(ix, ix))  # You'll need to pass width or calculate it
    return ix, iy

def map_to_world(ix, iy, origin_x, origin_y, resolution):
    """Convert map indices to world coordinates with +Y=North convention"""
    wx = origin_x + ix * resolution
    wy = origin_y + iy * resolution  # Direct conversion - no flipping needed
    return wx, wy

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
    def __lt__(self, other): return self.cost < other.cost

# Heuristic (Euclidean + angle)
def heuristic(node, goal, h_weight=1.1):
    dx = goal.x - node.x
    dy = goal.y - node.y
    dtheta = (goal.theta - node.theta + math.pi) % (2 * math.pi) - math.pi

    dist = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    angle_to_target = (target_angle - node.theta + math.pi) % (2 * math.pi) - math.pi
    turn_radius = 10.0

    if abs(angle_to_target) < 0.1:
        return dist + abs(dtheta) * turn_radius * h_weight
    else:
        turn_cost = abs(angle_to_target) * turn_radius
        final_turn_cost = abs(dtheta) * turn_radius
        return turn_cost + dist + final_turn_cost * h_weight

def proximity_penalty(dist_map, x, y):
    clearance = 10.0; w = 20.0
    # FIXED: Correct indexing for flipped coordinate system
    row = int(dist_map.shape[0] - 1 - y)  # Flip Y for distance map access
    col = int(x)
    if row < 0 or col < 0 or row >= dist_map.shape[0] or col >= dist_map.shape[1]:
        return w  # High penalty for out of bounds
    d = dist_map[row, col]
    return 0.0 if d >= clearance else w * (1.0 - d/clearance) ** 2

def is_collision(x, y, grid):
    ix, iy = int(x), int(y)
    # FIXED: Correct indexing for flipped coordinate system
    row = int(grid.shape[0] - 1 - iy)  # Flip Y for grid access
    col = ix
    if row < 0 or col < 0 or row >= grid.shape[0] or col >= grid.shape[1]:
        return True
    return grid[row, col] == 1

# A* planner
def a_star(start, goal, grid, dist_map, res):
    rospy.loginfo('Running A* path planning')
    ang_vel = 0.6
    turn_radius = 0.5/res
    swing_vel = turn_radius*ang_vel
    vel = 0.5 / res
    cost_scale = 1.0; back_penalty = 1.5; swing_penalty = 1.3
    actions = [
        (vel, 0.0, vel * cost_scale),
        (swing_vel, ang_vel, swing_vel * swing_penalty * cost_scale),
        (swing_vel, -ang_vel, swing_vel * swing_penalty * cost_scale),
        (-vel, 0.0, back_penalty * vel * cost_scale),
        (-swing_vel, ang_vel, back_penalty * swing_vel * swing_penalty * cost_scale),
        (-swing_vel, -ang_vel, back_penalty * swing_vel * swing_penalty * cost_scale)
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

        i += 1
        if i % 1000 == 0:
            best_dist = math.hypot(cur.x - goal.x, cur.y - goal.y)
            if i > 50000 and best_dist > initial_dist * 0.8:
                rospy.loginfo("Search stalled, increasing tolerances by 50%")
                tol_px *= 1.5
                tol_angle *= 1.5

        angle_diff = abs((cur.theta - goal.theta + math.pi) % (2 * math.pi) - math.pi)
        if abs(cur.x - goal.x) < tol_px and abs(cur.y - goal.y) < tol_px and angle_diff < tol_angle:
            rospy.loginfo('Found path after %d iterations', i)
            path = []
            action_path = []
            while cur:
                path.append((cur.x, cur.y, cur.theta))
                if cur.action:
                    action_path.append(cur.action)
                cur = cur.parent
            return path[::-1], action_path[::-1]

        for v, w, c in actions:
            new_theta = cur.theta + w
            new_theta = (new_theta + math.pi) % (2 * math.pi) - math.pi
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
    return [], []

# ROS node
class AStarPlannerNode(object):
    def __init__(self):
        rospy.init_node('astar_planner')

        # Load map
        map_yaml = rospy.get_param('~map_yaml')
        map_dir = os.path.dirname(map_yaml)
        m = read_yaml(map_yaml)
        yaml_img = m['image']
        pgm_path = os.path.join(map_dir, yaml_img)
        self.res = m['resolution']

        with open(pgm_path, 'rb') as f:
            raster = read_pgm(f)
        self.height, self.width = raster.shape

        # FIXED: Flip the grid vertically so bottom-left becomes origin
        self.grid = np.flipud(threshold_map(raster))  # Flip upside down
        self.dist_map = distance_transform_edt(self.grid == 0)

        self.origin_x, self.origin_y, self.origin_th = m['origin']

        # Transform matrices (simplified - you may need to adjust based on your specific needs)
        self.T_wo = np.array([[1.0,0.0,-self.origin_x/self.res],
                              [0.0,1.0,-self.origin_y/self.res],  # Removed the flip here
                              [0,0,1]])
        self.T_oi = np.eye(3)
        self.T_ir = np.eye(3)

        # Publishers and subscribers
        self.path_pub = rospy.Publisher('planned_path', Path, queue_size=1)
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('move_base_simple/goal', PoseStamped, self.goal_cb)
        rospy.loginfo('A* planner ready with +Y=North coordinate system')
        rospy.spin()

    def goal_cb(self, msg):
        # Get start pose in world coords
        amcl = rospy.wait_for_message('amcl_pose', PoseWithCovarianceStamped)
        wx_s = amcl.pose.pose.position.x
        wy_s = amcl.pose.pose.position.y
        _, _, th_s = tf.transformations.euler_from_quaternion([
            amcl.pose.pose.orientation.x,
            amcl.pose.pose.orientation.y,
            amcl.pose.pose.orientation.z,
            amcl.pose.pose.orientation.w])

        wx_g = msg.pose.position.x
        wy_g = msg.pose.position.y
        _, _, th_g = tf.transformations.euler_from_quaternion([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w])

        # FIXED: Convert to map indices with new coordinate system
        ix_s, iy_s = world_to_map(wx_s, wy_s, self.origin_x, self.origin_y, self.res, self.height)
        ix_g, iy_g = world_to_map(wx_g, wy_g, self.origin_x, self.origin_y, self.res, self.height)

        start = Node(ix_s, iy_s, th_s, 0, None, None)
        goal = Node(ix_g, iy_g, th_g, 0, None, None)
        curr = Node(ix_s, iy_s, th_s, 0, None, None)

        rospy.loginfo("[AMCL & msg] start_raw=(%.3f, %.3f, %.2f) goal_raw=(%.3f, %.3f, %.2f)",
                      wx_s, wy_s, th_s, wx_g, wy_g, th_g)
        rospy.loginfo("[A*] start=(%d, %d, %.2f) goal=(%d, %d, %.2f)",
                      ix_s, iy_s, th_s, ix_g, iy_g, th_g)

        # MPC loop
        horizon = 4
        while (abs(curr.x-goal.x)>5 or abs(curr.y-goal.y)>5 or
               abs((curr.theta-goal.theta+math.pi)%(2*math.pi)-math.pi)>0.5):

            _, path_actions = a_star(curr, goal, self.grid, self.dist_map, self.res)
            if not path_actions:
                rospy.logwarn("No path found!")
                break

            path_actions_mpc = path_actions[:horizon]
            rospy.loginfo("[MPCA*] actions planned: %s", path_actions_mpc)

            for a in path_actions_mpc:
                twist = Twist()
                twist.linear.x = a[0] * self.res
                twist.angular.z = -a[1]  # Note: may need sign adjustment based on your robot
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

            rospy.loginfo("[Odom] Current=(%.3f, %.3f, %.2f)", wx_c, wy_c, th_c)

            # Convert back to map coordinates
            ix_c, iy_c = world_to_map(wx_c, wy_c, self.origin_x, self.origin_y, self.res, self.height)
            rospy.loginfo("[MPCA*] Current=(%d, %d, %.2f)", ix_c, iy_c, th_c)
            curr = Node(ix_c, iy_c, th_c, 0, None, None)

        rospy.loginfo("[MPCA*] Done.")

if __name__ == '__main__':
    try:
        AStarPlannerNode()
    except rospy.ROSInterruptException:
        pass