# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for processing motion clips."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np

from  CrossLoco.utils.motion_utilities import pose3d 
from pybullet_utils import transformations
import pybullet


def standardize_quaternion(q):
  """Returns a quaternion where q.w >= 0 to remove redundancy due to q = -q.

  Args:
    q: A quaternion to be standardized.

  Returns:
    A quaternion with q.w >= 0.

  """
  if q[-1] < 0:
    q = -q
  return q


def normalize_rotation_angle(theta):
  """Returns a rotation angle normalized between [-pi, pi].

  Args:
    theta: angle of rotation (radians).

  Returns:
    An angle of rotation normalized between [-pi, pi].

  """
  norm_theta = theta
  if np.abs(norm_theta) > np.pi:
    norm_theta = np.fmod(norm_theta, 2 * np.pi)
    if norm_theta >= 0:
      norm_theta += -2 * np.pi
    else:
      norm_theta += 2 * np.pi

  return norm_theta


def calc_heading(q):
  """Returns the heading of a rotation q, specified as a quaternion.

  The heading represents the rotational component of q along the vertical
  axis (z axis).

  Args:
    q: A quaternion that the heading is to be computed from.

  Returns:
    An angle representing the rotation about the z axis.

  """
  ref_dir = np.array([1, 0, 0])
  rot_dir = pose3d.QuaternionRotatePoint(ref_dir, q)
  heading = np.arctan2(rot_dir[1], rot_dir[0])
  return heading


def calc_heading_rot(q):
  """Return a quaternion representing the heading rotation of q along the vertical axis (z axis).

  Args:
    q: A quaternion that the heading is to be computed from.

  Returns:
    A quaternion representing the rotation about the z axis.

  """
  heading = calc_heading(q)
  q_heading = transformations.quaternion_about_axis(heading, [0, 0, 1])
  return q_heading


def to_matrix(position, roll_pitch_yaw):
  cos_yaw, sin_yaw = np.cos(roll_pitch_yaw[2]), np.sin(roll_pitch_yaw[2])
  return np.array(((cos_yaw, -sin_yaw, position[0]),
                   (sin_yaw, cos_yaw, position[1]),
                   (0.0, 0.0, 1.0)))


def normalize_angle(theta):
  return (theta + np.pi) % (2 * np.pi) - np.pi


def T_in_root(joint_name, pose):
    # get hand trajectory in the elbow coordinate
    skel = pose.skel
    joint = skel.get_joint(joint_name)
    T = np.dot(
        joint.xform_from_parent_joint,
        pose.data[skel.get_index_joint(joint)],
    )
    while joint.parent_joint.name != 'root':
        T_j = np.dot(
            joint.parent_joint.xform_from_parent_joint,
            pose.data[skel.get_index_joint(joint.parent_joint)],
        )
        T = np.dot(T_j, T)
        joint = joint.parent_joint
    return T


def set_pose(robot, pose):
  num_joints = pybullet.getNumJoints(robot)
  root_pos = pose[0:3]
  root_rot = pose[3:7]

  pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

  for j in range(num_joints):
    j_info = pybullet.getJointInfo(robot, j)
    j_state = pybullet.getJointStateMultiDof(robot, j)

    j_pose_idx = j_info[3]
    j_pose_size = len(j_state[0])
    j_vel_size = len(j_state[1])

    if (j_pose_size > 0):
      j_pose = pose[j_pose_idx:(j_pose_idx + j_pose_size)]
      j_vel = np.zeros(j_vel_size)
      pybullet.resetJointStateMultiDof(robot, j, j_pose, j_vel)

  return