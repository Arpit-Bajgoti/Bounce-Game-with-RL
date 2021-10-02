import pygame
from enum import Enum
from collections import namedtuple, deque
import time
import random
import tensorflow as tf
import numpy as np
import os

import torch

from model import Linear_QNet, QTrainer

# 1536 x 864 is the current screen resolution of my pc
pygame.init()
radius = 10
width = 836
height = 664
block_size = 20

# rgb colors
DARK_GREEN = (0, 100, 0)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
font = pygame.font.SysFont('arial', 25)
MAX_LEN = 100_000
counter = 0

n_episodes = 1001  # n games we want agent to play (default 1001)
output_dir = 'model_output/game/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class Direction(Enum):
    LEFT = 0
    RIGHT = 1


Point = namedtuple("Point", "w, h")
color_dict = {1: DARK_GREEN, 0: GREEN}


class TurboGame:
    def __init__(self, w=width, h=height):
        self.w = w
        self.h = h
        self.w_initial = int(0.25 * self.w)
        self.h_initial = int(0.9 * self.h)
        self.counter = 3
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Turbo")
        self.score = 0
        self.bat = None
        self.update_bat()
        self.direction = None
        self.ball = None
        self._place_ball()
        self.game_over = False
        self.update = [6, 6]
        self.radius = radius
        self.width = 120
        self.height = 30
        self.lst = []
        self.stone = []
        self.bat_length = 60
        self.counter = 5
        self.parameters = deque(maxlen=MAX_LEN)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.0995
        self.epsilon_min = 0.01
        self.reward = 0
        self.state = None
        self.batch_size = 100
        self.next_state = None
        self.direction_one_hot = [0, 0]
        self.model = Linear_QNet(2, 100, 2)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
        self.clock = pygame.time.Clock()

    def update_bat(self):
        self.bat = [Point(self.w_initial, self.h_initial),
                    Point(int(self.w * 0.2), int(0.01 * self.h))]

    def play_step(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self.update_bat()
        self.update_ball()
        self.direction = self.act(self.state)
        self._move(self.direction)
        self._update_ui()
        self.clock.tick(50)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return self.game_over, self.score, self.get_state()

    def _place_ball(self):
        x = random.randint(40, self.w)
        y = random.randint(100, int(0.5 * self.h_initial))
        self.ball = [x, y]

    def _update_ui(self):
        self.display.fill(BLACK)

        pygame.draw.rect(self.display, BLUE1,
                         pygame.Rect(self.bat[0].w, self.bat[0].h, self.bat[1].w, 30))

        pygame.draw.circle(self.display, RED, tuple(self.ball), self.radius)

        self.draw_brick()
        text = font.render("Score: " + str(self.score), True, MAGENTA)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        if direction == 0 and self.w_initial < (self.w - self.bat[1].w - 20):
            self.w_initial += block_size
        elif direction == 1 and self.w_initial > 10:
            self.w_initial -= block_size

    def update_ball(self):
        # ball hitting bat
        if (self.ball[1] + self.radius) > self.h_initial - 10:
            if (self.ball[0] - self.radius > self.w_initial - 10) and self.ball[0] + self.radius < self.w_initial + \
                    self.bat[1].w + 10:
                self.update[1] = random.randint(6, 10)
                self.update[1] = -self.update[1]
                self.score += 10
                self.reward = 15
            else:
                self.reward = -15
                self.game_over = True
        # ball hitting boundaries
        elif self.ball[0] + self.radius > self.w:
            self.update[0] = -self.update[0]
        elif self.ball[1] + self.radius < 20:
            self.update[1] = -self.update[1]
            self.update[1] = random.randint(5, 10)
        elif self.ball[0] + self.radius < 20:
            self.update[0] = -self.update[0]
            self.update[0] = random.randint(5, 10)
        self.stone_conditions()
        # updating ball at each while loop
        self.ball[0] += self.update[0]
        self.ball[1] += self.update[1]

    # TODO: Reset integrating DNN with the play_step

    def reset(self):
        self.score = 0
        # self.reward = 0
        self._place_ball()

    def brick(self):
        for i in range(30, self.w - self.width - 10, self.width + 10):
            for j in range(10, self.h - (self.height * 8), self.height + 10):
                x = i
                y = j
                z = [x, y]
                self.stone.append(z)

    def draw_brick(self):
        for value in self.lst:
            pygame.draw.rect(self.display, color_dict[value[2]],
                             pygame.Rect(value[0], value[1], self.width, self.height))

    def get_state(self):
        input_features = [self.ball[0], self.ball[1]]
        return input_features

    def remember(self):
        if self.direction:
            self.direction_one_hot = [0, 1]
        else:
            self.direction_one_hot = [1, 0]

        self.parameters.append((self.get_state(), self.direction_one_hot, self.reward,
                                self.next_state, self.game_over))

    def stone_conditions(self):
        for value in self.lst:
            if value[0] < (self.ball[0] - self.radius) < (value[0] + self.width) and \
                    value[1] + 20 > (
                    self.ball[1] - self.radius) > (value[1] - self.height + 10):
                self.update[1] = -self.update[1]
                x = self.lst.index(value)
                if value[2]:
                    value[2] -= 1
                    self.score += 5
                else:
                    self.score += 10
                    self.lst.pop(x)
                    self.counter -= 1

    def train_long_memory(self):
        if len(self.parameters) > self.batch_size:
            mini_sample = random.sample(self.parameters, self.batch_size)  # list of tuples
        else:
            mini_sample = self.parameters

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step_high(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step_low(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # if acting randomly, take random action
            return random.randrange(2)
        state = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state)
        self.direction = torch.argmax(prediction).item()
        return self.direction  # pick the action that will give the highest reward (i.e., go left or right?)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':
    game = TurboGame()
    # game loop
    game.brick()
    game.lst = random.sample(game.stone, random.randint(5, 8))
    for val in game.lst:
        val.append(random.randint(0, 1))
    while True:
        if game.counter == 3:
            num = random.randint(2, 4)
            add = random.sample(game.stone, num)
            for val in add:
                val.append(random.randint(0, 1))
            game.lst.extend(add)
            game.counter = num + 3
        # actual main

        game.game_over, score, game.state = game.play_step()
        game.next_state = game.state
        game.remember()
        game.train_short_memory(game.get_state(), game.direction_one_hot, game.reward,
                                game.next_state, game.game_over)

        if game.game_over:
            print(f"Final Score : {game.score}, Reward : {game.reward}")
            if len(game.parameters) > game.batch_size:
                game.train_long_memory()
            game.reset()
            game.game_over = False

    # pygame.quit()
