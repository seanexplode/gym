from gym import utils
from gym.envs.robotics import fetch_env
from random import uniform


class FetchBlockStacking(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        var = 0.25
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            # qpos[0~2] : root joint cartesian position (x, y, z) of center of mass
            # qpos[3~6] : orientation
            'object0:joint': [1.25 - uniform(-var, var), 0.7 - uniform(-var, var), 0.42, 1., 0., 0., 0.],
            'object1:joint': [1.25 - uniform(-var, var), 0.7 - uniform(-var, var), 0.42, 1., 0., 0., 0.],
            'object2:joint': [1.25 - uniform(-var, var), 0.7 - uniform(-var, var), 0.42, 1., 0., 0., 0.],
            'object3:joint': [1.25 - uniform(-var, var), 0.7 - uniform(-var, var), 0.42, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
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
