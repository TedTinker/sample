#%%

import pybullet as p
import numpy as np
import cv2
from random import uniform
import matplotlib.pyplot as plt

from utils import get_physics, image_size



class Arena():
    def __init__(self, GUI = False):
        self.arena_map = cv2.imread("map.png")
        self.w, self.h, _ = self.arena_map.shape
        self.physicsClient = get_physics(GUI, self.w, self.h)
        self.already_constructed = False

    def start(self):
        if(not self.already_constructed):
            p.loadURDF("plane.urdf", [0,0,0], globalScaling = 2,
                       useFixedBase = True, physicsClientId = self.physicsClient) 
            for loc in ((x,y) for x in range(self.w) for y in range(self.h)):
                pos = [loc[0],loc[1],.5]
                ors = p.getQuaternionFromEuler([0,0,0])
                color = self.arena_map[loc][::-1] / 255
                color = np.append(color, 1)
                cube = p.loadURDF("cube.urdf", (pos[0], pos[1], pos[2]), 
                                ors, globalScaling = 1, useFixedBase = True, physicsClientId = self.physicsClient)
                p.changeVisualShape(cube, -1, rgbaColor = color, physicsClientId = self.physicsClient)
            self.already_constructed = True

    def take_photo(self, show = False):
        x, y, z = uniform(0, self.w), uniform(0, self.h), uniform(1, 10)
        tx, ty, tz = uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [x, y, z], 
            cameraTargetPosition = [x - tx, y - ty, 0], 
            cameraUpVector = [0, 0, 1], physicsClientId = self.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = .01, 
            farVal = 100, physicsClientId = self.physicsClient)
        _, _, rgba, depth, _ = p.getCameraImage(
            width=image_size, height=image_size,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.physicsClient)
        rgba = np.divide(rgba, 255)
        rgb = rgba[:, :, :-1]
        
        if(show):
            plt.imshow(rgb)
            plt.axis('off')
            plt.show()
            plt.close()
        
        return(rgb)
    
    

if __name__ == "__main__":
    arena = Arena(GUI = True)
    arena.start()
    for i in range(100):
        arena.take_photo(show = True)
# %%
