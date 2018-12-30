# -*- coding: utf-8 -*-
"""
Toy Problem of Robotic Picking
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class JumpingCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_actiony = 0
        self.max_actionx = 1.0
        self.max_actiony = 1.0
        self.min_posx = -1.0
        self.min_posy = -0.5
        self.max_posx = 1.0
        self.max_posy = 0.5
        self.max_speedx = 1.
        self.max_speedy = 1.
        self.goal_posx = 0.45   # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_posy = -0.3   # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.power = 0.01

        self.low_state = np.array([self.min_posx, self.min_posy, -self.max_speedx, -self.max_speedy])
        self.high_state = np.array([self.max_posx, self.max_posy, self.max_speedx, self.max_speedy])

        self.viewer = None

        self.action_space = spaces.Box(low=np.array([-self.max_actionx, self.min_actiony]), 
                                            high=np.array([self.max_actionx, self.max_actiony]))
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        delta = 2e-3
        ground_delta = delta / 20
        #  posx = self.state[0]
        #  velx = self.state[1]
        posx, posy, velx, vely = self.state
        forcex = min(max(action[0], -1.0), 1.0)
        forcey = min(max(action[1], self.min_actiony), 1.0)

        # run this code only in case it was not in ground previous step
        ground = np.absolute(posy - self.min_posy) < ground_delta
        if ground:
            #  velx += force*self.power -0.0025 * math.cos(3*posx)
            velx += forcex*self.power
            if (velx > self.max_speedx): velx = self.max_speedx
            if (velx < -self.max_speedx): velx = -self.max_speedx

            # change vely
            vely += forcey*self.power

        vely -= 0.00024

        posx += velx
        posy += vely

        if (posx > self.max_posx): posx = self.max_posx
        if (posx < self.min_posx): posx = self.min_posx
        if (posx==self.min_posx and velx<0): velx = 0

        if (posy > self.max_posy): posy = self.max_posy
        if (posy < self.min_posy): posy = self.min_posy
        if (posy==self.min_posy and vely<0): vely = 0


        #  nearby_goal = np.absolute(posx - self.goal_posx) < delta and np.absolute(posy - self.goal_posy) < delta
        nearby_goal = ((posx - self.goal_posx) ** 2 + (posy - self.goal_posy) ** 2) < delta ** 2

        if nearby_goal:
            reward = 1.0
        else:
            reward = 0.0

        self.state = np.array([posx, posy, velx, vely])

        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), self.min_posy, 0, 0])
        #  self.state = np.array([self.goal_posx, self.goal_posy + 0.1, 0, 0])
        return np.array(self.state)

#    def get_state(self):
#        return self.state

    def _height(self, xs):
        #  return np.sin(3 * xs)*.45+.55
        if isinstance(xs, np.ndarray):
            return np.zeros(xs.shape)
        else:
            return 0

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 300

        world_width = self.max_posx - self.min_posx
        scale = screen_width/world_width
        carwidth=20
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_posx, self.max_posx, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_posx)*scale, ys*scale))

            # draw track
            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(5)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/4)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/4)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)

            # drawing target
            targetcenterx = (self.goal_posx-self.min_posx)*scale
            targetbottom = (self._height(self.goal_posx) + self.goal_posy - self.min_posy)*scale
            targetwidth = carwidth
            targetheight = carheight

            l = targetcenterx - targetwidth/2
            r = targetcenterx + targetwidth/2
            t = targetbottom + targetheight
            b = targetbottom
            print((l, r, t, b))
            target = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            target.set_color(0, 0, .5)
            self.viewer.add_geom(target)

            #  flagx = (self.goal_posx-self.min_posx)*scale
            #  flagy1 = self._height(self.goal_posx)*scale
            #  flagy2 = flagy1 + 50
            #  flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            #  self.viewer.add_geom(flagpole)
            #  flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            #  flag.set_color(.8,.8,0)
            #  self.viewer.add_geom(flag)

        posx, posy = self.state[0], self.state[1]
        self.cartrans.set_translation((posx-self.min_posx)*scale, (self._height(posx) + posy - self.min_posy)*scale)
        #  self.cartrans.set_rotation(math.cos(3 * posx))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
