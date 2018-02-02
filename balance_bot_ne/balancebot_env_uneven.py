import os
import math
import random

import logging
import numpy as np

import pybullet as p
from gym import spaces

from balance_bot_ne.balancebot_env_noise import BalancebotEnvNoise

logger = logging.getLogger(__name__)

class BalancebotEnvUneven(BalancebotEnvNoise):

    def __init__(self, vdrange=(-1, 1), render=False):

        super(BalancebotEnvUneven, self).__init__(render=render)

        # pitch, roll, pitch gyro, roll gyro, com.sp., req.sp.
        self.observation_space = [np.array([-math.pi, -math.pi,
                                            -math.pi, -math.pi, 
                                            -5, -5]),
                                  np.array([math.pi, math.pi,
                                            math.pi, math.pi, 
                                            5, 5])]
        self.vdrange = vdrange

    def _load_geometry(self):
        shiftX = np.random.uniform(-0.5,0.5)
        shiftY = np.random.uniform(-0.5,0.5)
        shift = [shiftX, shiftY, -0.15]
        meshScale=[0.15, 0.15, 0.1]
        path = os.path.abspath(os.path.dirname(__file__))

        self.groundColId = p.createCollisionShape(shapeType=p.GEOM_MESH, 
                                                  fileName=os.path.join(path, "ground.obj"), 
                                                  collisionFramePosition=shift,
                                                  meshScale=meshScale,
                                                  flags=p.GEOM_FORCE_CONCAVE_TRIMESH)

        self.groundId = p.createMultiBody(baseMass=0,
                                              baseInertialFramePosition=[0,0,0],
                                              baseCollisionShapeIndex=self.groundColId, 
                                              basePosition=[0,0,0], 
                                              useMaximalCoordinates=True)

    def _compute_observation(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        linear, angular = p.getBaseVelocity(self.botId)
        obs =  np.array([cubeEuler[0] + np.random.normal(0,0.05) + self.pitch_offset,
                cubeEuler[1] + np.random.normal(0,0.05),
                angular[2] + np.random.normal(0,0.01),
                angular[0] + np.random.normal(0,0.01),
                self.vt, self.vd])
        return obs

    def reset(self):
        self.pitch_offset = np.random.normal(0,0.1)
        super(BalancebotEnvNoise, self).reset()
        self.vd = random.uniform(*(self.vdrange)) # need to assign AFTER call to super
        return self._compute_observation() # call to update obs according to new vd

    def step(self, action):
        if random.random() < 0.01:
            self.vd = random.uniform(*(self.vdrange))
        return super(BalancebotEnvNoise, self).step(action)
