import numpy as np
import pygame
from typing import Dict, List

_gen_name_variable = 1


def generate_name():
    global _gen_name_variable
    name = f'NCar-ID{_gen_name_variable}'
    _gen_name_variable += 1
    return name


class Settings:
    screen_size = (800, 600)
    fps = 60

    speed_change = 5 / fps
    rotation_change = 2 / fps

    track_quality = 10
    track_input_rate = 5

    rotations = np.radians(np.arange(-90, 90.1, 5))
    # rotations = np.array((0,))

    min_speed = 0.0
    max_speed = 5

    car_size = [30, 45]
    car_image = pygame.transform.rotate(pygame.transform.scale(pygame.image.load('car1.png'), car_size), 90)


class Globals:
    screen = pygame.display.set_mode(Settings.screen_size)
    timer = pygame.time.Clock()

    epoch_number = 0

    start_pos = [0, 0]
    start_speed = 0.1
    start_rotation = 0

    car_num = 1

    track = [np.array([(0, 0), (0, 100), (100, 100), (100, 0)])]
    track_to_draw = track

    @staticmethod
    def draw_track():
        for line in Globals.track_to_draw:
            pygame.draw.aalines(Globals.screen, (255, 255, 255), True, line)

    @staticmethod
    def new_track():
        track = []

        a_pos = (0, 0)
        b_pos = (0, 0)

        mode = False  # True: drawing track; False: change first variables
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
        Globals.start_speed = min(max((w ** 2 + h ** 2) ** 0.5 / Settings.screen_size[1] / 2,
                                      Settings.min_speed), Settings.max_speed)
        Globals.start_rotation = np.arctan(h / w if w != 0 else 10 ** 7)
        Globals.track = track
        Globals.track_to_draw = track


def d(a, b):
    return a / b if b != 0 else 10 ** 7


def get_intersect_point(tang, pos1, pos2):  # vector: (1, np.tan(rotation)) - остался только np.tan; A: point; B: point
    return get_interpos([(0, 0), (1, tang)], [pos1, pos2])
    # h, w = (pos1[1] - pos2[1]), (pos1[0] - pos2[0])
    # f = h / w if w != 0 else 10 ** 7
    # # print(f, tang)
    # x = f * pos1[0] / (f - tang)
    # y = tang * x
    # return [x, y]


def get_interpos(l1, l2):
    f1, f2 = [d(l[0][1] - l[1][1], l[0][0] - l[1][0]) for l in (l1, l2)]
    if f1 == f2:
        return False
    px = (f1 * l1[0][0] - l1[0][1] - (f2 * l2[0][0] - l2[0][1])) / (f1 - f2)
    py = l1[0][1] + f1 * (px - l1[0][0])
    return [px, py]


class NeuralCar:
    def __init__(self, pos, speed, rotation):  # first variables
        self.ID = generate_name()
        self.alive = True
        self.pos = pos
        self.rotation = rotation
        self.speed = speed
        self.fitness = 0

    def get_variables(self):
        return {self.ID: {'is_alive': self.alive,  # type: bool
                          'distances': self.get_distances(),  # type: List[float]
                          'current_speed': self.speed,  # type: float
                          'fitness': self.fitness}}  # type: float

    def update_variables(self, accelerate, rotate):
        self.speed = min(max(self.speed + accelerate, Settings.min_speed), Settings.max_speed)
        rotate *= (self.speed / Settings.max_speed) ** 0.5
        self.rotation += rotate
        self.pos = [self.pos[0] + self.speed * np.cos(self.rotation), self.pos[1] + self.speed * np.sin(self.rotation)]
        self.fitness += self.speed

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
                        inter_pos = get_interpos([(0, 0), (1, tang)], [line[i - 1], line[i]])
                        if 1:
                            if tang * inter_pos[1] * r_s >= 0:
                                dist = min(dist, (inter_pos[0] ** 2 + inter_pos[1] ** 2) ** 0.5)
                            inter_pos = np.array(inter_pos, dtype=np.int64)
                            print(round(tang, 3), inter_pos, (tang * inter_pos[1]) / abs(tang * inter_pos[1]))
                            inter_pos = [inter_pos[0] + int(self.pos[0]), inter_pos[1] + int(self.pos[1])]
                            pygame.draw.circle(Globals.screen, (255, 0, 0), [int(i) for i in inter_pos], 5)
            distances.append(max(dist, 30))
        return distances


class Epoch:
    def __init__(self):
        self.cars = []
        self.cars_ID: Dict[str: NeuralCar] = {}
        self.new_epoch()

    def new_epoch(self):
        if Globals.epoch_number % Settings.track_input_rate == 0:
            Globals.new_track()
        Globals.epoch_number += 1
        self.cars: List[NeuralCar] = [NeuralCar(Globals.start_pos, Globals.start_speed, Globals.start_rotation)
                                      for _ in range(Globals.car_num)]
        self.cars_ID: Dict[str: NeuralCar] = {car.ID: car for car in self.cars}

    def tick(self):
        Globals.draw_track()
        for car in self.cars:
            car.draw()

    def get_car_variables(self):
        var_dict = {}
        for car in self.cars:
            var_dict.update(car.get_variables())
        return var_dict

    def update_car_variables(self, update_dict: dict):
        for car_ID in list(update_dict.keys()):
            acc, rot = update_dict[car_ID]
            self.cars_ID[car_ID].update_variables(acc, rot)
