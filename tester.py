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
                rot = -Settings.speed_change
            elif event.key == 276:
                rot = Settings.speed_change
        elif event.type == pygame.KEYUP:
            if event.key in (273, 274):
                acc = 0
            elif event.key in (275, 276):
                rot = 0
    variables = ep.get_car_variables()
    car = ep.cars[0]
    distances = variables[ep.cars[0].ID]['distances']
    variables[car.ID] = [acc, rot]
    ep.update_car_variables(variables)

    Globals.screen.fill((0, 0, 0))
    ep.tick()
    for r, dist in zip(car.rotation + Settings.rotations, distances):
        pos = [car.pos[0] + dist * np.cos(r), car.pos[1] + dist * np.sin(r)]
        pygame.draw.line(Globals.screen, (255, 255, 255), car.pos, pos, 3)
    Globals.timer.tick()


pygame.quit()
