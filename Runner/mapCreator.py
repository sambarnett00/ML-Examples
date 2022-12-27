import pygame
import pickle
from utils import MapInfo
from os import mkdir
from os.path import exists
from typing import Tuple, List
pygame.init()


"""
CONTROLS:
Left Mouse - place track
Right Mouse - remove track
0           - removes starting point
1           - place starting point
Mouse Button 4 - Drag and place ckpt (ckpts must be in order)
Delete - Removes the nearest ckpt
S - Save, prompt appears in terminal
L - Load, prompt appears in terminal 
R - Reset map
UP Arrow - Increase track size
DOWN Arrow - Decrease track size
Right Arrow - highlights ckpts in order
Left Arrow - highlights ckpts in reverse order
"""


FRAMEWIDTH, FRAMEHEIGHT = 1600, 900
# FRAMEWIDTH, FRAMEHEIGHT = 1920, 1080
window = pygame.display.set_mode((FRAMEWIDTH, FRAMEHEIGHT))
surf = pygame.Surface((FRAMEWIDTH, FRAMEHEIGHT))

PEN_RAD = 32
FILL_COLOUR = (0, 0, 0)

surf.fill(FILL_COLOUR)
info = MapInfo()
info.reset()
mprev = pygame.mouse.get_pressed(num_buttons=5)

xy = None
ckpt_highlight = -1
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): ## QUIT
            pygame.quit()
            quit()
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:  ## SAVE
                name = input("Name >> ")
                if exists(f"maps/{name}"):
                    print(f"{name} dir already exists")
                else:
                    if not exists("maps"):
                        mkdir("maps")
                    mkdir(name)
                    pygame.image.save(surf, f"maps/{name}/map.png")

                    with open(f"maps/{name}/map.data", "wb") as file:
                        pickle.dump(info, file)

            elif event.key == pygame.K_l:  ## LOAD
                name = input("Name >> ")
                if not exists(f"maps/{name}"):
                    print(f"{name} does not exist!")
                else:
                    img = pygame.image.load(f"maps/{name}/map.png")
                    if img.get_width() != FRAMEWIDTH or img.get_height() != FRAMEHEIGHT:
                        img = pygame.transform.scale(img, (FRAMEWIDTH, FRAMEHEIGHT))
                    surf.blit(img, (0, 0))

                    with open(f"maps/{name}/map.data", "rb") as file:
                        info = pickle.load(file)
            
            elif event.key == pygame.K_r:  ## RESET
                surf.fill(FILL_COLOUR)
                info.reset()
            
            elif event.key == pygame.K_DELETE:
                mpos = pygame.mouse.get_pos()
                mn: Tuple[int, float] = None
                for i, line in enumerate(info.ckpts):
                    mid = ((line[0] + line[2])/2, (line[1] + line[3])/2)
                    d = ((mid[0]-mpos[0])**2 + (mid[1]-mpos[1])**2)**0.5
                    if mn is None or mn[1] > d:
                        mn = (i, d)

                if mn is not None:
                    info.ckpts.pop(mn[0])
                
            elif event.key == pygame.K_0: info.start = None
            elif event.key == pygame.K_1: info.start = pygame.mouse.get_pos()
            elif event.key == pygame.K_UP: PEN_RAD += 2
            elif event.key == pygame.K_DOWN: PEN_RAD = max(2, PEN_RAD-2)
            elif event.key == pygame.K_LEFT: 
                if len(info.ckpts) <= 0: continue
                ckpt_highlight = (ckpt_highlight - 1) % len(info.ckpts)
            elif event.key == pygame.K_RIGHT:
                ckpt_highlight = (ckpt_highlight + 1) % len(info.ckpts)


    mpos, mpress = pygame.mouse.get_pos(), pygame.mouse.get_pressed(num_buttons=5)
    if mpress[0]:
        pygame.draw.circle(surf, [255, 255, 255], mpos, PEN_RAD) ## draw
    if mpress[2]:
        pygame.draw.circle(surf, FILL_COLOUR, mpos, PEN_RAD)  ## erase

    if mpress[4] and not mprev[4]: ## checkpoints
        xy = mpos
    elif not mpress[4] and mprev[4] and xy is not None:
        info.ckpts.append(xy + mpos)
        xy = None

    ## Render
    window.blit(surf, (0, 0))
    for i, line in enumerate(info.ckpts):
        col = [0, 255, 0] if i != ckpt_highlight else [255, 0, 0]
        pygame.draw.line(window, col, line[:2], line[2:], 2)

    if xy is not None:
        pygame.draw.line(window, [0, 255, 0], xy, mpos, 2)
    
    if info.start is not None:
        pygame.draw.circle(window, [255, 0, 0], info.start, 5)

    pygame.display.update()
    mprev = mpress
