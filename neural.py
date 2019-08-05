from images import *
import pygame
import numpy as np


class Neural:
    def __init__(self):
        pass

    def get(self, variables) -> list:  # vars: dict['is_alive', 'distances', 'current_speed', 'fitness' - keys]
        #                              return: list[accelerate, rotate]
        accelerate, rotate = 1, 0  # -1 <= value <= 1
        dists = variables['distances']
        r = 0.1
        if dists[0] - dists[-1] > 0.01:
            rotate = -1 + np.random.uniform(0, r)
        elif dists[0] - dists[-1] < -0.01:
            rotate = 1 + np.random.uniform(0, r)
        else:
            rotate = 0 + np.random.uniform(-r, r)
        return [accelerate * Settings.speed_change, rotate * Settings.rotation_change]


class Run:
    def __init__(self):
        self.running = False
        self.neural = Neural()
        self.epoch = Epoch()

    def run(self):
        self.running = True
        while self.running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
            variables = self.epoch.get_car_variables()
            variables = {ID: self.neural.get(variables[ID]) for ID in list(variables.keys()) if variables[ID]['alive']}
            self.epoch.update_car_variables(variables)
            self.epoch.draw()


Run().run()
