#!/usr/bin/env python
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
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, Quaternion, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Header

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
    return np.where(raster == 254, 0, 1)

class Node:
    def __init__(self, x, y, theta, cost, parent):
        self.x = x; self.y = y; self.theta = theta; self.cost = cost; self.parent = parent
    def __lt__(self, other): return self.cost < other.cost

def heuristic(n, g):
    return math.hypot(g.x-n.x, g.y-n.y) + abs(g.theta-n.theta)

def proximity_penalty(dist_map, x, y):
    clearance = 10.0; w = 20.0
    d = dist_map[int(x), int(y)]
    return 0.0 if d >= clearance else w * (1.0 - d/clearance) ** 2

def is_collision(x, y, grid):
    ix, iy = int(x), int(y)
    if ix < 0 or iy < 0 or ix >= grid.shape[0] or iy >= grid.shape[1]: return True
    return grid[ix, iy] == 1

def a_star(start, goal, grid, dist_map, res):
    tol_px = 0.25 / res
    ang_vel = 0.6; swing_vel = 0.2 / res; vel = 0.5 / res
    cost_scale = 1.0; back_penalty = 2.0; swing_penalty = 1.0
    actions = [
        (vel, 0.0, vel * 0.5 * cost_scale),
        (swing_vel, ang_vel, swing_vel * swing_penalty * cost_scale),
        (swing_vel, -ang_vel, swing_vel * swing_penalty * cost_scale),
        (-vel, 0.0, back_penalty * vel * 0.5 * cost_scale),
        (-swing_vel, ang_vel, back_penalty * swing_vel * swing_penalty * cost_scale),
        (-swing_vel, -ang_vel, back_penalty * swing_vel * swing_penalty * cost_scale)
    ]
    open_list = []
    heapq.heappush(open_list, (0, start))
    closed = set()
    while open_list:
        _, cur = heapq.heappop(open_list)
        key = (int(cur.x), int(cur.y), round(cur.theta, 2))
        if key in closed: continue
        closed.add(key)
        if abs(cur.x - goal.x) < tol_px and abs(cur.y - goal.y) < tol_px and abs(cur.theta - goal.theta) < 0.6:
            path = []
            while cur:
                path.append(cur)
                cur = cur.parent
            return path[::-1]
        for v, w, c in actions:
            new_theta = cur.theta + w
            if w == 0:
                dx = v * math.cos(cur.theta); dy = v * math.sin(cur.theta)
            else:
                dx = v / w * (math.sin(new_theta) - math.sin(cur.theta))
                dy = -v / w * (math.cos(new_theta) - math.cos(cur.theta))
            nx, ny = int(cur.x + dx), int(cur.y + dy)
            if not is_collision(nx, ny, grid):
                pen = proximity_penalty(dist_map, nx, ny)
                nc = cur.cost + c + pen
                node = Node(nx, ny, round(new_theta, 2), nc, cur)
                f = nc + heuristic(node, goal)
                heapq.heappush(open_list, (f, node))
    return []

class AStarPlannerNode(object):
    def __init__(self):
        rospy.init_node('astar_planner')
        map_yaml = rospy.get_param('~map_yaml', 'mymap.yaml')
        # resolve YAML directory
        map_dir = os.path.dirname(map_yaml)
        m = read_yaml(map_yaml)
        yaml_img = m['image']
        pgm_path = os.path.join(map_dir, yaml_img)
        self.res = m['resolution']
        with open(pgm_path, 'rb') as f: raster = read_pgm(f)
        self.grid = threshold_map(raster)
        self.dist_map = distance_transform_edt(self.grid == 0)
        self.path_pub = rospy.Publisher('planned_path', Path, queue_size=1)
        self.cmd_pub = rospy.Publisher('cmd_vel_actions', Twist, queue_size=1)
        rospy.Subscriber('move_base_simple/goal', PoseStamped, self.goal_cb)
        rospy.loginfo('A* planner ready')
        rospy.spin()

    def goal_cb(self, msg):
        # get current pose from amcl (PoseWithCovarianceStamped)
        amcl = rospy.wait_for_message('amcl_pose', PoseWithCovarianceStamped)
        sx = amcl.pose.pose.position.x / self.res
        sy = amcl.pose.pose.position.y / self.res
        # orientation to yaw
        import tf
        (_, _, sth) = tf.transformations.euler_from_quaternion([
            amcl.pose.pose.orientation.x,
            amcl.pose.pose.orientation.y,
            amcl.pose.pose.orientation.z,
            amcl.pose.pose.orientation.w])
        gx = msg.pose.position.x / self.res
        gy = msg.pose.position.y / self.res
        (_, _, gth) = tf.transformations.euler_from_quaternion([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w])
        start = Node(sx, sy, sth, 0, None)
        goal = Node(gx, gy, gth, 0, None)
        path_states = a_star(start, goal, self.grid, self.dist_map, self.res)
        # publish Path
        ros_path = Path(); ros_path.header = Header(frame_id='map')
        for n in path_states:
            ps = PoseStamped(); ps.header = ros_path.header
            ps.pose.position = Point(n.x * self.res, n.y * self.res, 0)
            q = tf.transformations.quaternion_from_euler(0, 0, n.theta)
            ps.pose.orientation = Quaternion(*q)
            ros_path.poses.append(ps)
        self.path_pub.publish(ros_path)
        # publish Twist actions
        for i in range(len(path_states)-1):
            cur = path_states[i]; nxt = path_states[i+1]
            dx = (nxt.x - cur.x) * self.res
            dy = (nxt.y - cur.y) * self.res
            theta = cur.theta
            twist = Twist()
            twist.linear.x = math.hypot(dx, dy)
            twist.angular.z = math.atan2(dy, dx) - theta
            self.cmd_pub.publish(twist)
            rospy.sleep(0.1)

if __name__ == '__main__':
    try:
        AStarPlannerNode()
    except rospy.ROSInterruptException:
        pass
