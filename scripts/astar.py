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

# Node class for A*
class Node:
    def __init__(self, x, y, theta, cost, parent, actions=None):
        if actions is None:
            actions = []
        self.x = x; self.y = y; self.theta = theta; self.cost = cost; self.parent = parent; self.actions = actions
    def __lt__(self, other): return self.cost < other.cost  # Defining comparision

# Heuristic (Euclidean + angle)
def heuristic(n, g):
    return math.hypot(g.x - n.x, g.y - n.y) + abs(g.theta - n.theta)

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
    ang_vel = 0.5
    swing_vel = 0.2 / res
    vel = 0.5 / res
    cost_scale = 1.0; back_penalty = 2.0; swing_penalty = 1.0
    actions = [
        (vel, 0.0, vel * 0.5 * cost_scale), # forward
        (swing_vel, ang_vel, swing_vel * swing_penalty * cost_scale), # Left forward
        (swing_vel, -ang_vel, swing_vel * swing_penalty * cost_scale), # Right forward
        (-vel, 0.0, back_penalty * vel * 0.5 * cost_scale), # backward
        (-swing_vel, ang_vel, back_penalty * swing_vel * swing_penalty * cost_scale), # Right backward
        (-swing_vel, -ang_vel, back_penalty * swing_vel * swing_penalty * cost_scale) # Left backward
    ]
    tol_px = 0.5 / res
    tol_angle = 0.6
    open_list = []
    heapq.heappush(open_list, (0, start))
    closed = set()
    max_iter = 200000
    i = 0
    while open_list and i < max_iter:
        i += 1
        _, cur = heapq.heappop(open_list)
        key = (cur.x//4, cur.y//4, (cur.theta*8)//(2*math.pi)%8)
        if key in closed:
            continue
        closed.add(key)
        # goal check
        if abs(cur.x - goal.x) < tol_px and abs(cur.y - goal.y) < tol_px and abs(cur.theta - goal.theta) < tol_angle:
            rospy.loginfo('Found path after %d iterations', i)
            path = []
            action_path = cur.actions
            while cur:
                path.append(cur)
                cur = cur.parent
            return path[::-1], action_path
        # expand
        for v, w, c in actions:
            new_theta = cur.theta + w
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
                node = Node(nx, ny, new_theta, nc, cur, cur.actions+[(v, w)])
                f = nc + heuristic(node, goal)
                heapq.heappush(open_list, (f, node))
    return []

# ROS node
class MPCAStarPlannerNode(object):
    def __init__(self):
        rospy.init_node('mpc_astar_planner')

        # MPC parameters
        self.mpc_horizon = 5  # Number of steps to execute before replanning
        self.control_rate = 5  # Hz for control loop
        self.replan_distance_threshold = 0.3  # meters
        self.replan_angle_threshold = 0.3  # radians

        # Current plan storage
        self.current_plan_states = []
        self.current_plan_actions = []
        self.plan_index = 0
        self.goal_pose = None
        self.executing_plan = False

        # Load map (same as before)
        map_yaml = rospy.get_param('~map_yaml')
        map_dir = os.path.dirname(map_yaml)
        m = read_yaml(map_yaml)
        yaml_img = m['image']
        pgm_path = os.path.join(map_dir, yaml_img)
        self.res = m['resolution']
        self.origin_not_set = True

        with open(pgm_path, 'rb') as f:
            raster = read_pgm(f)
        self.height, self.width = raster.shape
        self.grid = threshold_map(raster)
        self.dist_map = distance_transform_edt(self.grid == 0)

        # Publishers and subscribers
        self.path_pub = rospy.Publisher('planned_path', Path, queue_size=1)
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('move_base_simple/goal', PoseStamped, self.goal_cb)

        # Timer for MPC control loop
        self.control_timer = rospy.Timer(rospy.Duration(1.0/self.control_rate), self.control_loop)

        rospy.loginfo('MPC A* planner ready')
        rospy.spin()

    def goal_cb(self, msg):
        """Handle new goal - start MPC planning"""
        self.goal_pose = msg
        self.executing_plan = True
        self.plan_index = 0
        self.replan()

    def get_current_pose(self):
        """Get current robot pose"""
        try:
            if self.origin_not_set:
                amcl = rospy.wait_for_message('amcl_pose', PoseWithCovarianceStamped, timeout=1.0)
                pose_msg = amcl.pose.pose
                # Set origin on first call
                self.origin_x = pose_msg.position.x
                self.origin_y = pose_msg.position.y
                self.origin_th = 0.0
                self.origin_not_set = False
            else:
                odom = rospy.wait_for_message('odom_combined', PoseWithCovarianceStamped, timeout=1.0)
                pose_msg = odom.pose.pose

            wx = pose_msg.position.x
            wy = pose_msg.position.y
            _, _, th = tf.transformations.euler_from_quaternion([
                pose_msg.orientation.x,
                pose_msg.orientation.y,
                pose_msg.orientation.z,
                pose_msg.orientation.w])

            return wx, wy, th
        except rospy.ROSException:
            rospy.logwarn("Failed to get current pose")
            return None, None, None

    def replan(self):
        """Execute A* planning from current position to goal"""
        if not self.goal_pose:
            return False

        # Get current pose
        wx_s, wy_s, th_s = self.get_current_pose()
        if wx_s is None:
            return False

        # Goal pose
        wx_g = self.goal_pose.pose.position.x
        wy_g = self.goal_pose.pose.position.y
        _, _, th_g = tf.transformations.euler_from_quaternion([
            self.goal_pose.pose.orientation.x,
            self.goal_pose.pose.orientation.y,
            self.goal_pose.pose.orientation.z,
            self.goal_pose.pose.orientation.w])

        # Convert to map indices
        ix_s, iy_s = world_to_map(wx_s, wy_s, self.origin_x, self.origin_y, self.res, self.height)
        ix_g, iy_g = world_to_map(wx_g, wy_g, self.origin_x, self.origin_y, self.res, self.height)

        start = Node(ix_s, iy_s, th_s, 0, None)
        goal = Node(ix_g, iy_g, th_g, 0, None)

        rospy.loginfo("[MPC A*] Replanning: start=(%d,%d,%.2f) goal=(%d,%d,%.2f)",
                      ix_s, iy_s, th_s, ix_g, iy_g, th_g)

        # Run A* planner
        path_states, path_actions = a_star(start, goal, self.grid, self.dist_map, self.res)

        if not path_states:
            rospy.logwarn("[MPC A*] No path found!")
            return False

        # Store new plan
        self.current_plan_states = path_states
        self.current_plan_actions = path_actions
        self.plan_index = 0

        rospy.loginfo("[MPC A*] New plan with %d states", len(path_states))

        # Publish visualization
        self.publish_path_visualization()

        return True

    def publish_path_visualization(self):
        """Publish the planned path for visualization"""
        if not self.current_plan_states:
            return

        ros_path = Path()
        ros_path.header = Header(frame_id='map', stamp=rospy.Time.now())

        for n in self.current_plan_states:
            ps = PoseStamped()
            ps.header = ros_path.header
            wx = self.origin_x + n.x * self.res
            wy = self.origin_y + (self.height - 1 - n.y) * self.res
            ps.pose.position = Point(wx, wy, 0)
            quat = tf.transformations.quaternion_from_euler(0, 0, n.theta)
            ps.pose.orientation = Quaternion(*quat)
            ros_path.poses.append(ps)

        self.path_pub.publish(ros_path)

    def should_replan(self):
        """Check if we should replan based on position error or completion"""
        if not self.current_plan_states or self.plan_index >= len(self.current_plan_states):
            return True

        # Get current pose
        wx_curr, wy_curr, th_curr = self.get_current_pose()
        if wx_curr is None:
            return False

        # Check if we're close to goal
        goal_dist = math.sqrt((wx_curr - self.goal_pose.pose.position.x)**2 +
                              (wy_curr - self.goal_pose.pose.position.y)**2)
        if goal_dist < 0.2:  # 20cm threshold
            rospy.loginfo("[MPC A*] Goal reached!")
            self.executing_plan = False
            return False

        # Check distance from expected position
        if self.plan_index < len(self.current_plan_states):
            expected_state = self.current_plan_states[self.plan_index]
            wx_exp = self.origin_x + expected_state.x * self.res
            wy_exp = self.origin_y + (self.height - 1 - expected_state.y) * self.res

            pos_error = math.sqrt((wx_curr - wx_exp)**2 + (wy_curr - wy_exp)**2)
            angle_error = abs(th_curr - expected_state.theta)

            if pos_error > self.replan_distance_threshold or angle_error > self.replan_angle_threshold:
                rospy.loginfo("[MPC A*] Replanning due to error: pos=%.3f, angle=%.3f",
                              pos_error, angle_error)
                return True

        # Check if we've executed enough actions
        if self.plan_index >= self.mpc_horizon:
            rospy.loginfo("[MPC A*] Replanning after %d steps", self.mpc_horizon)
            return True

        return False

    def control_loop(self, event):
        """Main MPC control loop"""
        if not self.executing_plan or not self.goal_pose:
            # Send stop command
            self.cmd_pub.publish(Twist())
            return

        # Check if we should replan
        if self.should_replan():
            if not self.replan():
                # Stop if planning fails
                self.cmd_pub.publish(Twist())
                return

        # Execute current action
        if (self.plan_index < len(self.current_plan_actions) and
                self.plan_index < len(self.current_plan_states)):

            action = self.current_plan_actions[self.plan_index]

            # Create control command
            twist = Twist()
            twist.linear.x = action[0] * self.res * 1.14  # Your scaling factors
            twist.angular.z = -action[1] * 1.2

            self.cmd_pub.publish(twist)

            rospy.logdebug("[MPC A*] Executing action %d: v=%.3f, w=%.3f",
                           self.plan_index, twist.linear.x, twist.angular.z)

            self.plan_index += 1
        else:
            # No more actions, stop
            self.cmd_pub.publish(Twist())

    def shutdown(self):
        """Clean shutdown"""
        self.cmd_pub.publish(Twist())  # Stop robot
        rospy.loginfo("[MPC A*] Shutting down")

if __name__ == '__main__':
    try:
        planner = MPCAStarPlannerNode()
    except rospy.ROSInterruptException:
        pass
