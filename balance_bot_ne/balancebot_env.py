import os
import math
import numpy as np

import pybullet as p
import pybullet_data

class BalancebotEnv():

    def __init__(self, render=False):
        self._observation = []
        self.action_space = [np.array([-1]), np.array([1])]
        self.observation_size = [np.array([-math.pi, -math.pi, -5]), 
                                 np.array([math.pi, math.pi, 5])] # pitch, gyro, com.sp.

        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

    def step(self, action):
        self._assign_throttle(action)
        p.stepSimulation()
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()

        self._envStepCounter += 1

        return np.array(self._observation), reward, done

    def reset(self):
        # reset is called once at initialization of simulation
        self.vt = 0
        self.vd = 0
        self.maxV = 24.6 # 235RPM = 24,609142453 rad/sec
        self._envStepCounter = 0
        
        p.resetSimulation()
        p.setGravity(0,0,-10) # m/s^2
        p.setTimeStep(0.01) # sec

        self._load_geometry()
        self._load_bot()

        # you *have* to compute and return the observation from reset()
        return self._compute_observation()

    def observation_space_size(self):
        return self.observation_space[0].size

    def action_space_size(self):
        return self.action_space[0].size

    def _assign_throttle(self, deltav):
        vt = clamp(self.vt + deltav, -self.maxV, self.maxV)
        self.vt = vt

        p.setJointMotorControl2(bodyUniqueId=self.botId, 
                                jointIndex=0, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=vt)
        p.setJointMotorControl2(bodyUniqueId=self.botId, 
                                jointIndex=1, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=-vt)

    def _load_geometry(self):
        self.groundId = p.loadURDF("plane.urdf")

    def _load_bot(self):
        cubeStartPos = [0,0,0.001]
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
        path = os.path.abspath(os.path.dirname(__file__))
        self.botId = p.loadURDF(os.path.join(path, "balancebot_simple.xml"),
                           cubeStartPos,
                           cubeStartOrientation)

    def _compute_observation(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        linear, angular = p.getBaseVelocity(self.botId)
        return np.array([cubeEuler[0],angular[0],self.vt])

    def _compute_reward(self):
        return 0.1 - abs(self.vt - self.vd) * 0.005

    def _compute_done(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        return abs(cubeEuler[0]) > 0.8 * math.pi/2 or cubePos[2] < -1 or self._envStepCounter >= 1000

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)
