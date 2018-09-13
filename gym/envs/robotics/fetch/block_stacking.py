from gym import utils
from gym.envs.robotics import fetch_env
from random import uniform


BLOCK_HEIGHT = 0.025


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class FetchBlockStacking(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        var = 0.22
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            # qpos[0~2] : root joint cartesian position (x, y, z) of center of mass
            # qpos[3~6] : orientation
            'object0:joint': [1.25 - uniform(-var, var), 0.8 - uniform(-var, var), 0.43, 1., 0., 0., 0.],
            'object1:joint': [1.25 - uniform(-var, var), 0.7 - uniform(-var, var), 0.43, 1., 0., 0., 0.],
            'object2:joint': [1.25 - uniform(-var, var), 0.7 - uniform(-var, var), 0.43, 1., 0., 0., 0.],
            'object3:joint': [1.25 - uniform(-var, var), 0.7 - uniform(-var, var), 0.43, 1., 0., 0., 0.],
            'object4:joint': [1.25 - uniform(-var, var), 0.7 - uniform(-var, var), 0.43, 1., 0., 0., 0.],
            'object5:joint': [1.25 - uniform(-var, var), 0.7 - uniform(-var, var), 0.43, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, 'fetch/block_stacking.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

        #  print("com of object 0", self.sim.data.get_site_xpos('object0'))
        #  print("com of object 1", self.sim.data.get_site_xpos('object1'))
        #  print("com of object 2", self.sim.data.get_site_xpos('object2'))

        print("body com of table", self.sim.data.get_body_xipos('table0'))

    def compute_reward(self, achieved_goal, goal, info):
        """Compute sparse reward in block stacking task"""
        global BLOCK_HEIGHT

        g = np.split(goal, 6)
        x = np.split(achieved_goal, 6)

        reward = 0
        for i in range(6):
            d = goal_distance(x[i], g[i])

            if d < delta:
                reward += 1
            else:
                break

        return reward


    # RobotEnv methods
    # ----------------------------

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        # object informations
        # object0
        object_pos_A = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot_A = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object_velp_A = self.sim.data.get_site_xvelp('object0') * dt
        object_velr_A = self.sim.data.get_site_xvelr('object0') * dt
        # gripper state
        object_rel_pos_A = object_pos_A - grip_pos
        object_velp_A -= grip_velp

        # object1
        object_pos_B = self.sim.data.get_site_xpos('object1')
        # rotations
        object_rot_B = rotations.mat2euler(self.sim.data.get_site_xmat('object1'))
        # velocities
        object_velp_B = self.sim.data.get_site_xvelp('object1') * dt
        object_velr_B = self.sim.data.get_site_xvelr('object1') * dt
        # gripper state
        object_rel_pos_B = object_pos_B - grip_pos
        object_velp_B -= grip_velp

        # object2
        object_pos_C = self.sim.data.get_site_xpos('object2')
        # rotations
        object_rot_C = rotations.mat2euler(self.sim.data.get_site_xmat('object2'))
        # velocities
        object_velp_C = self.sim.data.get_site_xvelp('object2') * dt
        object_velr_C = self.sim.data.get_site_xvelr('object2') * dt
        # gripper state
        object_rel_pos_C = object_pos_C - grip_pos
        object_velp_C -= grip_velp

        # object3
        object_pos_D = self.sim.data.get_site_xpos('object3')
        # rotations
        object_rot_D = rotations.mat2euler(self.sim.data.get_site_xmat('object3'))
        # velocities
        object_velp_D = self.sim.data.get_site_xvelp('object3') * dt
        object_velr_D = self.sim.data.get_site_xvelr('object3') * dt
        # gripper state
        object_rel_pos_D = object_pos_D - grip_pos
        object_velp_D -= grip_velp

        # object4
        object_pos_E = self.sim.data.get_site_xpos('object4')
        # rotations
        object_rot_E = rotations.mat2euler(self.sim.data.get_site_xmat('object4'))
        # velocities
        object_velp_E = self.sim.data.get_site_xvelp('object4') * dt
        object_velr_E = self.sim.data.get_site_xvelr('object4') * dt
        # gripper state
        object_rel_pos_E = object_pos_E - grip_pos
        object_velp_E -= grip_velp

        # object5
        object_pos_F = self.sim.data.get_site_xpos('object5')
        # rotations
        object_rot_F = rotations.mat2euler(self.sim.data.get_site_xmat('object5'))
        # velocities
        object_velp_F = self.sim.data.get_site_xvelp('object5') * dt
        object_velr_F = self.sim.data.get_site_xvelr('object5') * dt
        # gripper state
        object_rel_pos_F = object_pos_F - grip_pos
        object_velp_F -= grip_velp

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        object_pos = np.concatenate([
            object_pos_A, object_pos_B, object_pos_C,
            object_pos_D, object_pos_E, object_pos_F,
            ])

        object_rel_pos = np.concatenate([
            object_rel_pos_A, object_rel_pos_B, object_rel_pos_C,
            object_rel_pos_D, object_rel_pos_E, object_rel_pos_F,
            ])

        object_rot = np.concatenate([
            object_rot_A, object_rot_B, object_rot_C,
            object_rot_D, object_rot_E, object_rot_F,
            ])

        object_velp = np.concatenate([
            object_velp_A, object_velp_B, object_velp_C,
            object_velp_D, object_velp_E, object_velp_F,
            ])

        object_velr = np.concatenate([
            object_velr_A, object_velr_B, object_velr_C,
            object_velr_D, object_velr_E, object_velr_F,
            ])

        achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _sample_goal(self):
        """
        sample goal
        g_0 = x_0 and
        g_i = g_{i-1} + [0, 0, height of block]
        """
        global BLOCK_HEIGHT

        g = [None for _ in range(6)]
        g[0] = self.sim.data.get_site_xpos('object0')

        for i in range(1, 6):
            g[i] = g[i-1] + np.array([0, 0, BLOCK_HEIGHT])

        return np.concatenate(g)
