import pygame, numpy as np, time, random
pygame.init()

gamewidth, gameheight = 1920, 1080
screen = pygame.display.set_mode((gamewidth, gameheight), pygame.FULLSCREEN)
gravity, speed = 0, 500#0.020, 500
numOfBirds = 50


def text(surface, col, pos, size, text, font, center):
    largetext = pygame.font.SysFont(font, size)
    if center:
        textsurf = largetext.render(text, True, col)
        textrect = textsurf.get_rect()
        textrect.center = pos
        surface.blit(textsurf, textrect)
    else:
        textsurf = largetext.render(text, True, col)
        surface.blit(textsurf, pos)


class NeuralNetwork:
    def __init__(self, layout, weights=None):
        self.inputSize, self.hiddenSize, self.outputSize = layout

        if weights is None:
            self.weights = [np.random.randn(self.hiddenSize, self.inputSize)/2, 
                            np.random.randn(self.outputSize, self.hiddenSize)/2]
        else:
            self.weights = weights

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, inputs):
        z = np.dot(self.weights[0], inputs)
        hidResult = [self.sigmoid(res + 1) for res in z]
        z = np.dot(self.weights[1], hidResult)
        return [self.sigmoid(res + 1) for res in z]


class Bird:
    JUMP = 1
    def __init__(self, x, y, r, weights=None):
        self.x, self.y, self.r = x, y, r
        self.vy = 0
        self.collision = False
        self.fitness = 0

        self.NN = NeuralNetwork([2, 5, 2], weights)

    def activate(self, dt, pipe):
        if not self.collision:
            self.fitness += 0.25
            #self.move(dt)
            self.jump(pipe, dt)
            self.render()
            self.checkCollision()

    def jump(self, pipe, dt):
        inputs = [pipe.x-self.x, self.y - (pipe.y + pipe.gap/2)]
        #point = [pipe.x + pipe.w/2, pipe.y + pipe.gap/2]
        #distance = ((point[0] - self.x)**2 + (point[1] - self.y)**2)**0.5
        choice = self.NN.forward(inputs)
        if choice[0] > 0.5: 
            self.y -= Bird.JUMP*dt * choice[0] ##0.25
        if choice[1] > 0.5:
            self.y += Bird.JUMP*dt * choice[1]

    def move(self, dt):
        self.vy += gravity*dt

        self.y += self.vy

    def checkCollision(self):
        if self.y > gameheight - self.r:
            self.collision = True

    def render(self):
        pygame.draw.circle(screen, (255, 255, 255), [int(self.x), int(self.y)], self.r)


class Pipe:
    def __init__(self, x, y, gap, w):
        self.x, self.y, self.gap, self.w = x, y, gap, w
        self.speed, self.add = 0.7, False
        self.gap = 300

    def active(self, birds, pipeArray, dt):
        pipeArray = self.move(pipeArray, dt)
        self.collision(birds)
        return pipeArray

    def move(self, pipeArray, dt):
        self.x -= self.speed*dt

        if self.x + self.w < 0:
            pipeArray.remove(self)

        if self.x < gamewidth - gamewidth/3 and not self.add:
            gap = random.randint(200, 400)
            pipeArray.append(Pipe(gamewidth, random.randint(0, gameheight - gap), gap, 100))
            self.add = True

        return pipeArray

    def collision(self, birds):
        for bird in birds:
            bx, by, bw, bh = bird.x - bird.r, bird.y - bird.r, bird.r*2, bird.r*2
            if self.x < bx + bw and bx < self.x + self.w and by < self.y:
                bird.collision = True

            if self.x < bx + bw and self.y + self.gap < by + bh and bx < self.x + self.w:
                bird.collision = True

    def render(self):
        pygame.draw.rect(screen, (255, 255, 255), [self.x, 0, self.w, self.y])
        pygame.draw.rect(screen, (255, 255, 255), [self.x, self.y + self.gap, self.w, gameheight - self.y - self.gap])


def main():
    MUTATION = 0.95
    genepool, bestBird = [], []
    generation, highscore = 0, 0
    while True:
        if bestBird:
            for i in range(len(genepool)):
                genepool.append(bestBird)

        if not genepool:
            birds = [Bird(int(gamewidth/4), int(gameheight/2), 25) for i in range(numOfBirds)]
        else:
            birds = []
            for i in range(numOfBirds):
                #idxs = np.random.choice(list(range(numOfBirds), p=props)
                parentA = random.choice(genepool)
                parentB = random.choice(genepool)
                child = []

                for wa, wb in zip(parentA, parentB):
                    child.append([])
                    for node1, node2 in zip(wa, wb):
                        child[-1].append([])
                        for wn1, wn2 in zip(node1, node2):
                            chance = random.random()
                            if chance > MUTATION: ##0.95:
                                child[-1][-1].append(random.random()*2 -1)
                            else:
                                child[-1][-1].append(random.choice([wn1, wn2]))

                birds.append(Bird(int(gamewidth/4), int(gameheight/2), 25, child))

        staticgap = random.randint(200, 400)
        #random.seed(3)
        pipeArray = [Pipe(gamewidth, random.randint(0, gameheight - staticgap), staticgap, 100) for i in range(1)]
        oldtime, dt = 0, 1
        #  score = 0
        while True:
            if oldtime == 0:
                oldtime = time.time()
                newtime = oldtime
            else:
                oldtime = newtime
                newtime = time.time()
                dt = (newtime - oldtime) * speed

            screen.fill((0, 0, 0))
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()

                    if event.key == pygame.K_s:
                        if bestBird != []:
                            np.save("BestBird", bestBird)

                    if event.key == pygame.K_l:
                        birds.append(Bird(int(gamewidth/4), int(gameheight/2), 25, np.load("bestBird.npy")))

            for pipe in pipeArray:
                pipeArray = pipe.active(birds, pipeArray, dt)

            for pipe in pipeArray:
                pipe.render()

            collide, alive = True, 0
            for bird in birds:
                pipe = pipeArray[0]
                if bird.x > pipe.x + pipe.w:
                    pipe = pipeArray[1]

                bird.activate(dt, pipe)
                if not bird.collision:
                    collide = False
                    alive += 1

            if collide:
                break

            mostfit = birds[0].fitness
            for bird in birds[1:]:
                if bird.fitness > mostfit:
                    mostfit = bird.fitness
                    if mostfit > highscore:
                        highscore = mostfit
                        bestBird = bird.NN.weights

            text(screen, (100, 100, 100), (25, 25), 25, "Fitness: {}".format(int(mostfit)), "comicsansms", False)
            text(screen, (100, 100, 100), (25, 50), 25, "HighScore: {}".format(int(highscore)), "comicsansms", False)
            text(screen, (100, 100, 100), (25, 75), 25, "Birds alive: {}".format(alive), "comicsansms", False)
            text(screen, (100, 100, 100), (25, 100), 25, "Generation: {}".format(generation), "comicsansms", False)
            #  text(screen, (100, 100, 100), (int(gamewidth/2), 150), 75, str(score), "comicsansms", True)

            pygame.display.update()

        genepool = []
        #scores = [bird.fitness for bird in birds]
        #totalScore = sum(scores)
        #props = [s/totalScore for s in scores]
        
        for bird in birds:
            if bird.fitness > 200:
                repeat = int(bird.fitness/2)

            elif bird.y < 0:
                repeat = 0
            elif bird.y > gameheight - 30:
                repeat = 1
            else:
                repeat = int(bird.fitness/20)

            for i in range(repeat):
                genepool.append(bird.NN.weights)

        generation += 1


if __name__ == "__main__":
    while True:
        main()
