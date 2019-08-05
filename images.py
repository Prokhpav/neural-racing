import numpy as np
import pygame
from typing import Dict, List

DEBUG = True
_gen_name_variable = 1


def generate_name():
    global _gen_name_variable
    name = f'NCar-ID{_gen_name_variable}'
    _gen_name_variable += 1
    return name


class Settings:
    screen_size = (800, 600)
    fps = 60

    scale = min(screen_size[0] / 800, screen_size[1] / 600)

    speed_change = 5 / fps
    rotation_change = 6 / fps

    track_quality = 15
    track_input_rate = 5

    rotations = np.radians(np.arange(-60, 60.1, 15))
    # rotations = np.array((0,))

    min_speed = 1 * scale
    max_speed = 5 * scale

    car_size = [30, 45]
    car_image = pygame.transform.rotate(pygame.transform.scale(pygame.image.load('car1.png'), car_size), 90)


class Globals:
    screen = pygame.display.set_mode(Settings.screen_size)
    timer = pygame.time.Clock()

    epoch_number = 0

    start_pos = [0, 0]
    start_speed = 1
    start_rotation = 0

    car_num = 1

    track = [np.array([(0, 0), (0, 100), (100, 100), (100, 0)])]
    track_to_draw = track
    track_scale = Settings.screen_size[0]

    @staticmethod
    def set_track(track: List[np.ndarray]):  # track: list[np.array[point(x, y), ...], ...]
        if not track:
            return
        min_pos, max_pos = [track[0][0, 0], track[0][0, 1]], [track[0][0, 0], track[0][0, 1]]
        for line in track:
            for i in range(2):
                min_pos[i] = min(min_pos[i], line[:, i].min())
                max_pos[i] = max(max_pos[i], line[:, i].max())
        # print(track)
        # print(min_pos, max_pos)
        size = [max_pos[i] - min_pos[i] for i in range(2)]
        track = [(line - min_pos) / size for line in track]
        Globals.track = track
        scale = size[0] / Settings.screen_size[0], size[1] / Settings.screen_size[1]
        i = (scale[0] < scale[1])
        Globals.track_scale = Settings.screen_size[i]
        Globals.track_to_draw = [line * Settings.screen_size[i] for line in track]

    @staticmethod
    def draw_track():
        for line in Globals.track_to_draw:
            pygame.draw.aalines(Globals.screen, (255, 255, 255), True, line)

    @staticmethod
    def new_track():
        track = []

        a_pos = (0, 0)
        b_pos = (0, 0)

        mode = True  # True: drawing track; False: change first variables
        mousedown = False
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    mode = not mode
                    if event.key == 13 and a_pos != (0, 0):
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mousedown = True
                        if mode:
                            track.append([event.pos])
                        else:
                            a_pos = event.pos
                            b_pos = event.pos
                    elif event.button == 3 and mode and track:
                        track = track[:-1]
                elif event.type == pygame.MOUSEMOTION and mousedown:
                    if mode:
                        if ((track[-1][-1][0] - event.pos[0]) ** 2 + (
                                track[-1][-1][1] - event.pos[1]) ** 2) ** 0.5 > Settings.track_quality:
                            track[-1].append(event.pos)
                    else:
                        b_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    mousedown = False
                    if mode:
                        track[-1].append(track[-1][0])
                        track[-1] = np.array(track[-1])
            Globals.screen.fill((0, 0, 0))
            if track:
                for line in track:
                    if len(line) >= 2:
                        pygame.draw.aalines(Globals.screen, (255, 255, 255), False, line)
            pygame.draw.aaline(Globals.screen, (255, 64, 32), a_pos, b_pos)
            pygame.draw.circle(Globals.screen, (255, 64, 32), a_pos, 5)
            pygame.display.update()
            Globals.timer.tick(Settings.fps)

        Globals.start_pos = a_pos
        w, h = (a_pos[0] - b_pos[0]), (a_pos[1] - b_pos[1])
        Globals.start_speed = min(max((w ** 2 + h ** 2) ** 0.5 / Settings.screen_size[1] * 2 * Settings.max_speed,
                                      Settings.min_speed), Settings.min_speed)
        Globals.start_rotation = np.arctan(h / w if w != 0 else 10 ** 7)
        if w >= 0:
            Globals.start_rotation += np.pi
        # Globals.set_track(track)
        Globals.track = track
        Globals.track_to_draw = track


# Globals.set_track(track=Globals.track)


def get_intersection_point(tang, pos1, pos2):  # l1 = [(0, 0), (1, tang)]
    f2 = (pos1[1] - pos2[1]) / (pos1[0] - pos2[0]) if pos1[0] != pos2[0] else 10 ** 7
    if f2 == tang:
        return False
    px = (f2 * pos1[0] - pos1[1]) / (f2 - tang)
    py = tang * px
    return [px, py]


class NeuralCar:
    def __init__(self, pos, speed, rotation):  # first variables
        self.ID = generate_name()
        self.alive = True
        self.time_alive = 0
        self.pos = pos
        self.rotation = rotation
        self.speed = speed
        self.fitness = 0

    def get_variables(self):
        return {'alive': self.alive,  # type: bool
                'distances': self.get_distances(),  # type: List[float]
                'current_speed': self.speed,  # type: float
                'fitness': self.fitness}  # type: float

    def update_variables(self, accelerate, rotate):
        self.speed = min(max(self.speed + accelerate, Settings.min_speed), Settings.max_speed)
        rotate *= (self.speed / Settings.max_speed) ** 0.5
        self.rotation += rotate
        self.pos = [self.pos[0] + self.speed * np.cos(self.rotation), self.pos[1] + self.speed * np.sin(self.rotation)]
        self.fitness += self.speed
        self.time_alive += 1 / Settings.fps

    def draw(self):
        image = pygame.transform.rotate(Settings.car_image, -np.degrees(self.rotation))
        Globals.screen.blit(image, [pos - im_size / 2 for pos, im_size in zip(self.pos, image.get_size())])
        pygame.draw.circle(Globals.screen, (255, 0, 0), [int(i) for i in self.pos], 5)

    def get_distances(self) -> List[float]:
        distances = []
        for rot in ((Settings.rotations + self.rotation) / np.pi) % 2:
            track = [line - self.pos for line in Globals.track]
            dist = 10000
            tang = np.tan(rot * np.pi)
            r_s = (-1 if 0.5 < rot < 1.5 else 1)
            for line in track:
                point = line[0]
                upper = (point[1] >= tang * point[0])
                for i in range(1, len(line)):
                    point = line[i]  
                    if upper != (point[1] >= tang * point[0]):
                        upper = not upper
                        inter_pos = get_intersection_point(tang, line[i - 1], line[i])
                        if inter_pos and tang * inter_pos[1] * r_s >= 0:
                            dist = min(dist, (inter_pos[0] ** 2 + inter_pos[1] ** 2) ** 0.5)
                            if DEBUG:
                                inter_pos = np.array(inter_pos, dtype=np.int64)
                                inter_pos = [inter_pos[0] + int(self.pos[0]), inter_pos[1] + int(self.pos[1])]
                                pygame.draw.circle(Globals.screen, (255, 0, 0), [int(i) for i in inter_pos], 5)
            distances.append(dist)
        return np.array(distances)


class Epoch:
    def __init__(self):
        self.cars = []
        self.dead_cars = []
        self.cars_ID: Dict[str: NeuralCar] = {}
        self.new_epoch()

    def new_epoch(self):
        if Globals.epoch_number % Settings.track_input_rate == 0:
            Globals.new_track()
        Globals.epoch_number += 1
        self.cars: List[NeuralCar] = [NeuralCar(Globals.start_pos, Globals.start_speed, Globals.start_rotation)
                                      for _ in range(Globals.car_num)]
        self.dead_cars.clear()
        self.cars_ID: Dict[str: NeuralCar] = {car.ID: car for car in self.cars}

    def tick(self):
        Globals.draw_track()
        for car in self.cars:
            car.draw()

    def draw(self):
        Globals.screen.fill((0, 0, 0))
        Globals.draw_track()
        self.tick()
        pygame.display.update()
        Globals.timer.tick(Settings.fps)

    def get_car_variables(self):
        var_dict = {}
        for car in self.cars:
            variables = car.get_variables()
            if np.any(variables['distances'] < Settings.car_size[0] / 2):
                variables['alive'] = False
                self.kill_car(car)
            variables['distances'] = variables['distances'] / Settings.screen_size[0]
            variables['current_speed'] = variables['current_speed'] / Settings.screen_size[0]
            var_dict[car.ID] = variables
        return var_dict

    def update_car_variables(self, update_dict: dict):
        if self.cars:
            for car_ID in list(update_dict.keys()):
                acc, rot = update_dict[car_ID]
                self.cars_ID[car_ID].update_variables(acc, rot)
        else:
            self.new_epoch()

    def kill_car(self, car):
        car.alive = False
        self.cars.remove(car)
        print(self.cars)
        self.cars_ID.pop(car.ID)
        self.dead_cars.append(car)
