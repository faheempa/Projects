import torch
import random
import numpy as np
from collections import deque
from game import *
from model import *
from model import Linear_QNet, QTrainer

MAX_MEMORY = 250_000
BATCH_SIZE = 1500
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.no_of_game = 0
        self.epsilon = 0
        self.gamma = 0.9  # discount rate (must be smaller than 1)
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(
            13, 512, 3
        )  # size of state, a random value, size of the action list
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # we are using a list to represent the state
        # [danger_straight(3), current_direction(4), food_location(4)]

        # finding points near to the head
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        # directions
        dir_l = Direction.LEFT
        dir_r = Direction.RIGHT
        dir_u = Direction.UP
        dir_d = Direction.DOWN
        # snake current direction
        sdir_l = game.direction == dir_l
        sdir_r = game.direction == dir_r
        sdir_u = game.direction == dir_u
        sdir_d = game.direction == dir_d

        state = [
            # danger straight
            (sdir_l and game.is_collision(point_l))
            or (sdir_r and game.is_collision(point_r))
            or (sdir_u and game.is_collision(point_u))
            or (sdir_d and game.is_collision(point_d)),
            # danger straight to the end
            # (sdir_l and self.tail_in(point_l, dir_l, game))
            # or (sdir_r and self.tail_in(point_r, dir_r, game))
            # or (sdir_u and self.tail_in(point_u, dir_u, game))
            # or (sdir_d and self.tail_in(point_d, dir_d, game)),
            # danger right
            (sdir_l and game.is_collision(point_u))
            or (sdir_r and game.is_collision(point_d))
            or (sdir_u and game.is_collision(point_r))
            or (sdir_d and game.is_collision(point_l)),
            # danger right to the end
            (sdir_l and self.tail_in(point_u, dir_u, game))
            or (sdir_r and self.tail_in(point_d, dir_d, game))
            or (sdir_u and self.tail_in(point_r, dir_r, game))
            or (sdir_d and self.tail_in(point_l, dir_l, game)),
            # danger left
            (sdir_l and game.is_collision(point_d))
            or (sdir_r and game.is_collision(point_u))
            or (sdir_u and game.is_collision(point_l))
            or (sdir_d and game.is_collision(point_r)),
            # danger left to the end
            (sdir_l and self.tail_in(point_d, dir_d, game))
            or (sdir_r and self.tail_in(point_u, dir_u, game))
            or (sdir_u and self.tail_in(point_l, dir_l, game))
            or (sdir_d and self.tail_in(point_r, dir_r, game)),
            # current direction
            sdir_r,
            sdir_d,
            sdir_l,
            sdir_u,
            # food side
            game.food.x > game.head.x,  # food left
            game.food.x < game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]
        return np.array(state, dtype=int)

    def tail_in(self, point, dir, game):
        if dir == Direction.RIGHT:
            while point.x + BLOCK_SIZE < game.w:
                point = Point(point.x + BLOCK_SIZE, point.y)
                if point in game.snake[2:]:
                    return True
        elif dir == Direction.LEFT:
            while 1 < point.x - BLOCK_SIZE:
                point = Point(point.x - BLOCK_SIZE, point.y)
                if point in game.snake[2:]:
                    return True
        elif dir == Direction.DOWN:
            while point.y + BLOCK_SIZE < game.h:
                point = Point(point.x, point.y + BLOCK_SIZE)
                if point in game.snake[2:]:
                    return True
        elif dir == Direction.UP:
            while 0 < point.y - BLOCK_SIZE:
                point = Point(point.x, point.y - BLOCK_SIZE)
                if point in game.snake[2:]:
                    return True
        return False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 200 - self.no_of_game
        final_move = [0, 0, 0]
        if random.randint(0, 400) < self.epsilon:
            # random moves
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # model prediction
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)
        if state_old[2]:
            print(state_old, " right")
        if state_old[4]:
            print(state_old, " left")

        # get move
        final_move = agent.get_action(state_old)

        # perform the action and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            agent.no_of_game += 1
            agent.train_long_memory()
            # update results
            if score > record:
                record = score
                agent.model.save()

            game.reset(top=record)


if __name__ == "__main__":
    train()
