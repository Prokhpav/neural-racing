from images import *
import pygame

ep = Epoch()

acc = 0
rot = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == 273:
                acc = Settings.speed_change
            elif event.key == 274:
                acc = -Settings.speed_change
            elif event.key == 275:
                rot = Settings.speed_change
            elif event.key == 276:
                rot = -Settings.speed_change
        elif event.type == pygame.KEYUP:
            if event.key in (273, 274):
                acc = 0
            elif event.key in (275, 276):
                rot = 0

    Globals.screen.fill((0, 0, 0))

    variables = ep.get_car_variables()
    car = ep.cars[0]
    distances = variables[car.ID]['distances']
    for ID in list(variables.keys()):
        variables[ID] = [0, 0]
    variables[car.ID] = [acc, rot]

    if DEBUG:
        for r, dist in zip(car.rotation + Settings.rotations, distances):
            pos = [car.pos[0] + dist * np.cos(r), car.pos[1] + dist * np.sin(r)]
            if np.pi * 0.5 < r % (np.pi * 2) < np.pi * 1.5:
                color = (255, 127, 127)
            else:
                color = (255, 255, 255)
            pygame.draw.line(Globals.screen, color, car.pos, pos, 3)

    ep.update_car_variables(variables)
    ep.tick()
    pygame.display.update()
    Globals.timer.tick(Settings.fps)


pygame.quit()
