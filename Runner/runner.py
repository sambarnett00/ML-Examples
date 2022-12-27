import pygame
import pickle
import numpy as np
from math import radians, sin, cos
from random import randint
from utils import MapInfo, line_circle, closest_dist
from networks import Network, NumpyNetwork
from time import time
from typing import Tuple, List


class Agent:
    RADIUS = 24
    SPEED = 1

    ROT_OFFSETS = (-45, 0, 45) ## direction of ray projection
    RAY_LENGTH = 384
    RAY_STEPS = 24

    NUM_FEATURES = len(ROT_OFFSETS)
    NUM_OUT = 2
    LAYOUT = [(NUM_FEATURES, None), (6, "relu"), (NUM_OUT, "tanh")]

    def __init__(self, map_info: MapInfo, map_img: pygame.Surface, brain: Network) -> None:
        self.x, self.y = map_info.start[0]*Simulation.MAP_SCALE, map_info.start[1]*Simulation.MAP_SCALE
        self.vel, self.rot = 0, 0
        self.fitness = 0
        self.alive = True

        self.brain = brain 
        self.map_img = map_img
        self.map_info = map_info
        self.next_ckpt = 0

    def tick(self, dt) -> None:
        if self.alive:
            # self.control()
            self.see()
            self.move(dt)
            self.collision()
            self.score()

    def control(self) -> None:
        ACC = 0.25
        keys = pygame.key.get_pressed()
        self.rot += keys[pygame.K_a] * -ACC + keys[pygame.K_d] * ACC
        self.vel += keys[pygame.K_w] * -ACC + keys[pygame.K_s] * ACC

    def see(self) -> None:
        """Sends ray inputs to Neural Network brain, and adds the logits to the velocity and rotation of the agent"""
        inputs = self.get_inputs()
        logits = self.brain.forward(inputs).reshape((Agent.NUM_OUT, )) ## vec(NUM_OUT, )
        
        self.vel += logits[0] * Agent.SPEED
        self.rot += logits[1] * Agent.SPEED

    def ray(self, rot: float, max_steps: int) -> float:
        """A Simple raycasting algo, samples the map image until collision is detected"""
        nx, ny = self.x, self.y
        fac = Agent.RAY_LENGTH / Agent.RAY_STEPS
        dx, dy = fac * cos(radians(rot+self.rot)), fac * sin(radians(rot+self.rot))
        for k in range(max_steps):
            col = self.map_img.get_at_mapped((int(nx), int(ny)))
            if not col: return k / max_steps

            nx += dx
            ny += dy

        return 1.0

    def move(self, dt: float) -> None:
        self.x += self.vel * cos(radians(self.rot)) * dt
        self.y += self.vel * sin(radians(self.rot)) * dt

    def collision(self) -> None:
        col = self.map_img.get_at_mapped((int(self.x), int(self.y)))
        if not col:
            self.alive = False

    def score(self) -> None: 
        """Updates the fitness of an agent when it has passed through the next checkpoint"""
        line = self.map_info.ckpts[self.next_ckpt]
        if line_circle(line, (self.x, self.y, Agent.RADIUS)):
            self.next_ckpt = (self.next_ckpt + 1) % len(self.map_info.ckpts)
            self.fitness += 1


    def render(self, xoff: float, yoff: float, render_rays=False) -> None:
        mx, my = self.x-xoff+Simulation.FRAMEWIDTH/2, self.y-yoff+Simulation.FRAMEHEIGHT/2
        if render_rays:
            for roff in Agent.ROT_OFFSETS:
                nx = mx + Agent.RAY_LENGTH * cos(radians(self.rot+roff))
                ny = my + Agent.RAY_LENGTH * sin(radians(self.rot+roff))
                pygame.draw.line(Simulation.WINDOW, (255, 0, 0), (mx, my), (nx, ny), 2)

        col = (0, 255, 0) if self.alive else (255, 0, 0)
        pygame.draw.circle(Simulation.WINDOW, col, (mx, my), Agent.RADIUS)
    
    def render_mini(self, surf: pygame.Surface, is_focus: bool) -> None:
        """Render Agent onto the mini-map"""
        col = (0, 255, 0) if self.alive else (255, 0, 0)
        if is_focus: col = (255, 225, 0)
        scale = Simulation.MINI_SCALE/Simulation.MAP_SCALE
        pygame.draw.circle(surf, col, (self.x*scale, self.y*scale), Agent.RADIUS*Simulation.MINI_SCALE)


    def get_inputs(self) -> np.ndarray:
        """Returns the inputs to the Neural Network, the sampled rays"""
        return np.array([[self.ray(roff, Agent.RAY_STEPS) for roff in Agent.ROT_OFFSETS]])

    def get_fitness(self) -> float:
        d = closest_dist(self.map_info.ckpts[self.next_ckpt], (self.x, self.y, Agent.RADIUS))
        return max(0, self.fitness + (1 - d/5000.0)) ## number of checkpoints passed + distances to next checkpoint scaled to <1
    
    def get_weights(self) -> Network:
        return self.brain.get_weights()


class Simulation:
    BRAIN_TYPE: Network = NumpyNetwork
    FRAMEWIDTH, FRAMEHEIGHT = 1600, 900
    MAP_SCALE, MINI_SCALE = 5, 0.2  ## MAP_SCALE is used to upscale the whole map, applied to FRAMEWIDTH&HEIGHT
    WINDOW: pygame.Surface
    SWITCH_DELAY = 0.25  ## seconds, delay between focus on 1 agent to the next

    NUM_AGENTS = 50 #1
    MUT_RATE = 0.05 # 0.03
    INIT_TIME = 15 #seconds
    FPS_COUNTER = False

    def __init__(self, map_name: str) -> None:
        Simulation.WINDOW = pygame.display.set_mode((Simulation.FRAMEWIDTH, Simulation.FRAMEHEIGHT))
        pygame.display.set_caption("Runner")

        self.map_info, raw_map_img = self.load_map(f"maps/{map_name}")
        self.map_info.scale(Simulation.MAP_SCALE)
        self.map_img, self.mini_map_img = self.load_images(raw_map_img)

        self.agents = [self.new_np_agent() for _ in range(Simulation.NUM_AGENTS)]
        self.alive_count = Simulation.NUM_AGENTS
        self.render_best, self.switch_delay = None, 0

        self.gen_highscore = 0
        self.highscore = 0
        self.generation = 1
        self.timer = Simulation.INIT_TIME
        self.pause = False

    def evolve(self):
        """ The genetic algorithm resides here
            Softmax is used to generate probabilities for each agent based on their fitness
            These probabilities determine how likely an agent is to be picked as a parent
            2 parents are picked for each agent in the next generation and their Neural Networks are combined
            Mutations are added (based on the MUT_RATE) to the weights of the NN
        """
        mx = max(agent.get_fitness() for agent in self.agents)
        bias_fn = lambda x: np.exp((x-mx+1))
        scores = np.array([bias_fn(agent.get_fitness()) for agent in self.agents])
        probs = scores / scores.sum()

        # best_idx = scores.argmax()
        # new_agents = [self.new_np_agent(self.agents[best_idx].get_brain())]
        new_agents = []
        index_range = list(range(Simulation.NUM_AGENTS))
        for _ in range(Simulation.NUM_AGENTS): # -1
            idxs = np.random.choice(index_range, p=probs, size=(2,))
            parent_a, parent_b = self.agents[idxs[0]], self.agents[idxs[1]]

            brain_c = [self.combine_matrix(wa, wb) for wa, wb in zip(parent_a.get_weights(), parent_b.get_weights())]
            new_agents.append(self.new_np_agent(NumpyNetwork(Agent.LAYOUT, brain_c)))
            
        self.agents = new_agents

    def combine_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ The best way to combine two matrices to facilitate evolution is up to debate
            A random point is selected, partitioning the matrices so that c = a[:i] + b[i:]
            Mutations are applied element-wise on the weights of the Neural Network
        """
        idx = randint(0, a.size)
        c = np.zeros((a.size, ))
        c[:idx] = a.reshape((a.size, ))[:idx]
        c[idx:] = b.reshape((b.size, ))[idx:]
        c = c.reshape(a.shape)
        
        mask = np.random.rand(*a.shape) < Simulation.MUT_RATE
        return c * ~mask +  c * mask * np.random.randn(*a.shape)
        #return c * ~mask + np.random.randn(*a.shape) * mask

    def new_np_agent(self, _brain: NumpyNetwork = None) -> Agent:
        brain = NumpyNetwork(Agent.LAYOUT, None) if _brain is None else _brain
        return Agent(self.map_info, self.map_img, brain)

    def run(self) -> None:
        Simulation.print_controls()
        t1 = time()
        while True:
            dt, t1 = (time()-t1+0.001), time()

            ## Tick
            self.handle_events()
            if not self.pause:
                best = self.tick_agents(dt)
                self.set_render_best(best, dt)
                self.tick_timer(dt)

            ## Render
            focus = self.get_focus()
            self.render_background(focus)
            self.render_agents(focus)
            self.render_details(focus, dt)
            self.render_mini_map(focus)
            pygame.display.update()

    @staticmethod
    def print_controls() -> None:
        with open("controls.txt", "r") as file:
            print(file.read())

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    self.pause = not self.pause
                elif event.key == pygame.K_RETURN:
                    self.end()
                elif event.key == pygame.K_RIGHT:
                    self.timer += 10

    def tick_agents(self, dt: float) -> Agent:
        self.alive_count = 0
        best = (-1, None)
        for agent in self.agents:
            agent.tick(dt)

            if agent.alive:
                self.alive_count += 1
                if (ftns := agent.get_fitness()) > best[0]:
                    best = (ftns, agent)
        
        if self.alive_count == 0:
            self.end()
        else:
            ftns = best[1].get_fitness()
            if ftns > self.gen_highscore:
                self.timer = Simulation.INIT_TIME ## if the score is increasing, do not start timer
                self.gen_highscore = ftns
            if ftns > self.highscore:
                self.highscore = ftns

        return best[1]

    def set_render_best(self, best: Agent, dt: float) -> None:
        if self.render_best is best: return

        self.switch_delay -= dt
        if self.switch_delay <= 0:
            self.render_best = best
            self.switch_delay = Simulation.SWITCH_DELAY
            
    def get_focus(self) -> Agent:
        assert len(self.agents) > 0, "Cannot focus on an Agent if there are none"
        if self.render_best is None:
            return self.agents[0]
        else:
            return self.render_best

    def tick_timer(self, dt: float) -> None:
        self.timer -= dt
        if self.timer < 0:
            self.end()

    def end(self) -> None:
        """This method is called when a generation has ended / forced to end"""
        self.timer = Simulation.INIT_TIME + self.generation / 2 ## every gen add 0.5s to timer
        self.generation += 1
        self.gen_highscore = 0
        self.evolve()


    # Render Methods
    def render_background(self, focus: Agent) -> None:
        xoff = -focus.x+Simulation.FRAMEWIDTH/2
        yoff = -focus.y+Simulation.FRAMEHEIGHT/2
        Simulation.WINDOW.fill((0, 0, 0))
        Simulation.WINDOW.blit(self.map_img, (xoff, yoff))

        x1, y1, x2, y2 = self.map_info.ckpts[focus.next_ckpt]
        pygame.draw.line(Simulation.WINDOW, (0, 0, 255), 
            (x1+xoff, y1+yoff), (x2+xoff, y2+yoff), 3)

    def render_agents(self, focus: Agent) -> None:
        for agent in self.agents:
            agent.render(focus.x, focus.y, render_rays=(agent is focus))

    def render_details(self, focus: Agent, dt: float) -> None:
        BUF, SIZE = 4, 20
        info = [
            f"Score: {focus.get_fitness():.2f}", f"Score: {self.highscore:.2f}",
            f"Alive: {self.alive_count}", f"Generation: {self.generation}",
            f"Timer: {self.timer:.2f}"]

        if Simulation.FPS_COUNTER: 
            info.append(f"FPS: {1.0/dt:.2f}")
        
        for i, txt in enumerate(info):
            self.render_text((164, 164, 164), (BUF, BUF + i*(BUF+SIZE)), SIZE, txt, "Courier New")

    def render_text(self, col: Tuple[int, int, int], pos: Tuple[int, int], size: int, text: str, font: str):
        largetext = pygame.font.SysFont(font, size)
        textsurf = largetext.render(text, True, col)
        Simulation.WINDOW.blit(textsurf, pos)

    def render_mini_map(self, focus: Agent) -> None:
        BUFFER = 8
        surf = self.mini_map_img.copy()
        for agent in self.agents:
            agent.render_mini(surf, agent is focus)

        x1, y1, x2, y2 = self.map_info.ckpts[focus.next_ckpt]
        scale = Simulation.MINI_SCALE/Simulation.MAP_SCALE
        pygame.draw.line(surf, (0, 0, 255), 
            (x1*scale, y1*scale), (x2*scale, y2*scale), 3)
        Simulation.WINDOW.blit(surf, (Simulation.FRAMEWIDTH-surf.get_width()-BUFFER, BUFFER))

    # Methods used to load images and data
    def load_map(self, dirname: str) -> Tuple[MapInfo, pygame.Surface]:
        with open(f"{dirname}/map.data", "rb") as file:
            map_info = pickle.load(file)
        
        map_img = pygame.image.load(f"{dirname}/map.png").convert()
        return map_info, map_img
    
    def load_images(self, raw_map_img: pygame.Surface) -> Tuple[pygame.Surface, pygame.Surface]:
        map_img = pygame.transform.scale(raw_map_img, 
            (Simulation.FRAMEWIDTH * Simulation.MAP_SCALE, Simulation.FRAMEHEIGHT * Simulation.MAP_SCALE))

        mini_map_img = pygame.transform.scale(raw_map_img,
            (Simulation.FRAMEWIDTH * Simulation.MINI_SCALE, Simulation.FRAMEHEIGHT * Simulation.MINI_SCALE)).convert_alpha()
        mini_map_img.set_alpha(127)
        grey = pygame.Surface(mini_map_img.get_size(), pygame.SRCALPHA, 32)
        grey.fill([127, 127, 127, 100])
        mini_map_img.blit(grey, [0, 0])
        ## pygame.draw.rect(mini_map_img, [127, 127, 127, 35], mini_map_img.get_rect())
        return map_img, mini_map_img


if __name__ == "__main__":
    pygame.init()
    Simulation("hard").run()