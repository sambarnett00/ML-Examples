import pygame
import numpy as np
from random import randint
from time import time
from statistics import NormalDist
from os.path import exists
from typing import List, Tuple


class Pipe:
    SPEED = 0.7

    def __init__(self, x: float, y: float, gap: float, w: float):
        self.x, self.y, self.gap, self.w = x, y, gap, w
        self.affiliate = False

        self.img = pygame.image.load("flappybirdpipe.png").convert_alpha()
        self.imgdown = pygame.transform.scale(self.img, (self.w, Simulation.FRAMEHEIGHT-gap))
        self.imgup = pygame.transform.flip(self.imgdown, False, True)

    def getInfo(self) -> Tuple[float, float, float, float]:
        return self.x, self.y, self.gap, self.w

    def tick(self, dt: float) -> bool:
        self.move(dt)
        self.render()
        return self.x + self.w < 0, self.x > Simulation.BIRD_XOFFSET ## yrem, coll

    def move(self, dt: float):
        self.x -= self.SPEED * dt

    def render(self):
        Simulation.WINDOW.blit(self.imgup, (self.x, self.y-self.imgup.get_height()))
        Simulation.WINDOW.blit(self.imgdown, (self.x, self.y+self.gap))
        # pygame.draw.rect(Simulation.WINDOW, (255, 255, 255), [self.x, 0, self.w, self.y])
        # pygame.draw.rect(Simulation.WINDOW, (255, 255, 255), [self.x, self.y + self.gap, self.w, Simulation.FRAMEHEIGHT - self.y - self.gap])


class NeuralNetwork:
    def __init__(self, layout: List[int], weights: List[np.ndarray] = None):
        self.layout = layout
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [np.random.randn(l1, l2) for l1, l2 in zip(layout[:-1], layout[1:])]

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for W in self.weights:
            x = NeuralNetwork.sigmoid(x @ W)
        
        return x


class Bird:
    JUMP = 1
    LAYOUT = (2, 5, 2)

    def __init__(self, x: float, y: float, r: float, weights: List[np.ndarray] = None):
        self.x, self.y, self.r = x, y, r
        self.collision = False
        self.fitness = 0

        self.NN = NeuralNetwork(Bird.LAYOUT, weights)
        self.img = pygame.image.load("flappybirdimg.png").convert_alpha()
        self.img = pygame.transform.scale(self.img, (int(self.r*2), int(self.r*2)))

    def tick(self, dt: float, pipe: Pipe):
        if not self.collision:
            self.fitness += 1
            self.jump(dt, pipe)
            self.render()
            self.checkCollision(pipe)

        return not self.collision

    def jump(self, dt: float, pipe: Pipe):
        px, py, _, _ = pipe.getInfo()
        inputs = np.array([[px - self.x, py - self.y]])
        logits = self.NN.forward(inputs)

        if logits[0][0] > 0.5: 
            self.y -= Bird.JUMP*dt * logits[0][0]
        if logits[0][1] > 0.5:
            self.y += Bird.JUMP*dt * logits[0][1]

    def render(self):
        Simulation.WINDOW.blit(self.img, (self.x-self.r, self.y-self.r))

    def checkCollision(self, pipe: Pipe):
        px, py, pg, pw = pipe.getInfo()
        if self.y + self.r > Simulation.FRAMEHEIGHT:
            self.collision = True
        
        if px < self.x + self.r and self.x - self.r < px + pw and (self.y - self.r < py or py + pg < self.y + self.r):
            self.collision = True
    
    def getWeights(self):
        return self.NN.weights


class Simulation:
    P_MUTATION = 0.05
    FRAMEWIDTH, FRAMEHEIGHT = 1920, 1080
    SPEED = 500
    NUM_BIRDS = 50
    BIRD_XOFFSET = FRAMEWIDTH // 4
    WINDOW: pygame.Surface = None

    def __init__(self):
        Simulation.WINDOW = pygame.display.set_mode((Simulation.FRAMEWIDTH, Simulation.FRAMEHEIGHT), pygame.FULLSCREEN)

        self.generation = 1
        self.score = 0
        self.highscore = 0
        self.aliveCount = 0
        self.oldPipe = None

        self.birds = [self.newBird() for _ in range(Simulation.NUM_BIRDS)]
        self.pipes = [self.newPipe()]
        self.best = None

        self.mut_zvalue = NormalDist(mu=0, sigma=1).inv_cdf(1-Simulation.P_MUTATION)

    def run(self):
        t1 = time()
        while True:
            dt, t1 = (time() - t1) * Simulation.SPEED, time()

            Simulation.WINDOW.fill((113, 197, 208))
            self.handleEvents()

            nextPipe = self.tickPipes(dt)
            self.aliveCount = self.tickBirds(dt, nextPipe)

            self.updateScore(nextPipe)
            self.renderDetails()

            pygame.display.update()

            if self.aliveCount == 0:
                self.reset()

    def handleEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

                elif event.key == pygame.K_s:
                    np.save("best.npy", self.best.getWeights(), allow_pickle=True)
                elif event.key == pygame.K_l:
                    if exists("best.npy"):
                        self.birds.append(self.newBird(np.load("best.npy", allow_pickle=True)))
                    else:
                        print("No best bird available")

    def tickPipes(self, dt: float) -> Pipe:
        nextPipe = None
        trim = False
        for pipe in self.pipes:
            yrem, coll = pipe.tick(dt)

            if yrem: trim = True
            if coll and nextPipe is None: nextPipe = pipe

        if trim:
            self.pipes.pop(0)

        if self.pipes[-1].getInfo()[0] < 0.66 * Simulation.FRAMEWIDTH and not self.pipes[-1].affiliate:
            self.pipes[-1].affiliate = True
            self.pipes.append(self.newPipe())

        return nextPipe

    def tickBirds(self, dt: float, nextPipe: Pipe) -> int:
        aliveCount = 0
        for bird in self.birds:
            if bird.tick(dt, nextPipe):
                aliveCount += 1
                self.best = bird

        return aliveCount

    def updateScore(self, nextPipe: Pipe):
        if self.oldPipe is None and nextPipe is not None:
            self.oldPipe = nextPipe

        if nextPipe is not self.oldPipe:
            self.score += 1
            self.highscore = max(self.highscore, self.score)
            self.oldPipe = nextPipe

    def renderDetails(self):
        self.renderText((100, 100, 100), (25, 25), 25, f"Score       {self.score}", "Courier New")
        self.renderText((100, 100, 100), (25, 50), 25, f"HighScore   {self.highscore}", "Courier New")
        self.renderText((100, 100, 100), (25, 75), 25, f"Birds alive {self.aliveCount}", "Courier New")
        self.renderText((100, 100, 100), (25, 100), 25, f"Generation  {self.generation}", "Courier New")

    def renderText(self, col: Tuple[int, int, int], pos: Tuple[int, int], size: int, text: str, font: str):
        largetext = pygame.font.SysFont(font, size)
        textsurf = largetext.render(text, True, col)
        Simulation.WINDOW.blit(textsurf, pos)

    def reset(self):
        self.generation += 1
        self.score = 0
        self.oldPipe = None
        self.pipes = [self.newPipe()]
        self.best = None
        self.nextGen()

    def nextGen(self):
        next_gen = []
        scores = np.array([bird.fitness for bird in self.birds])
        props = scores / scores.sum()

        bestIdx = scores.argmax() ## best bird from previous generation carries over
        next_gen.append(self.newBird(self.birds[bestIdx].getWeights()))

        indexRange = list(range(len(self.birds)))
        for _ in range(Simulation.NUM_BIRDS-1):
            idxs = np.random.choice(indexRange, p=props, size=(2, ))
            brainA, brainB = self.birds[idxs[0]].getWeights(), self.birds[idxs[1]].getWeights()

            child = [self.combineNN(wa, wb) for wa, wb in zip(brainA, brainB)]
            next_gen.append(self.newBird(child))

        self.birds = next_gen

    def combineNN(self, wa: np.ndarray, wb: np.ndarray) -> np.ndarray:
        weights = np.zeros_like(wa) ## same as wb, note: ndim=2
        for i in range(wa.shape[1]):
            weights.T[i] = wa.T[i] if np.random.random() > 0.5 else wb.T[i] ## merging step
        
        muts = np.random.randn(*wa.shape)
        mask = abs(muts) > self.mut_zvalue
        return weights * ~mask + muts * mask ## combine weights and mutations

    def newBird(self, weights: np.ndarray = None) -> Bird:
        return Bird(Simulation.FRAMEWIDTH//4, Simulation.FRAMEHEIGHT//2, 25, weights=weights)
    
    def newPipe(self) -> Pipe:
        gap = 300 ## todo: dynamic gap size?
        return Pipe(Simulation.FRAMEWIDTH, randint(0, Simulation.FRAMEHEIGHT-gap), gap, 100)


if __name__ == "__main__":
    pygame.init()
    Simulation().run()
