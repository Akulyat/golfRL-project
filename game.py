import pygame
import numpy as np
import math
import json

SIZE = 1000

class Car:
	def __init__(self):
		self.position = [SIZE // 2, SIZE // 2]
		self.direction = 0 # degrees
		self.velocity = 0
		self.acceleration = 0
		self.fuel = 1

	def update(self, obstacles):
		self.velocity += self.acceleration
		self.acceleration *= 0.8
		self.velocity *= 0.9
		if self.acceleration < 0.01:
			self.acceleration = 0
		if self.velocity < 0.01:
			self.velocity = 0
		addX = self.velocity * math.cos(self.direction)
		addY = self.velocity * math.sin(self.direction)
		newX = self.position[0] + addX
		newY = self.position[1] + addY
		for obstacle in obstacles:
			if not obstacle.x <= self.position[0] <= obstacle.x + obstacle.w and obstacle.x <= newX <= obstacle.x + obstacle.w and obstacle.y <= newY <= obstacle.y + obstacle.h:
				if self.position[0] < obstacle.x:
					newX = obstacle.x - (addX - (obstacle.x - self.position[0]))
				else:
					newX = obstacle.x + obstacle.w - (addX - (self.position[0] - obstacle.x - obstacle.w))
				self.direction = math.pi - self.direction
			if not obstacle.y <= self.position[1] <= obstacle.y + obstacle.h and obstacle.y <= newY <= obstacle.y + obstacle.h and obstacle.x <= newX <= obstacle.x + obstacle.w:
				if self.position[1] < obstacle.y:
					newY = obstacle.y - (addY - (obstacle.y - self.position[1]))
				else:
					newY = obstacle.y + obstacle.h - (addY - (self.position[1] - obstacle.y - obstacle.h))
				self.direction = -self.direction
		self.position[0] = newX
		self.position[1] = newY

class Obstacle:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.w = 80
		self.h = 80

class Hole:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.r = 10

	def check(self, ball):
		return (ball.position[0] - self.x) ** 2 + (ball.position[1] - self.y) ** 2 <= self.r ** 2 and ball.velocity < 1

	# def drive(self, move):
	# 	if move.startswith('R'):
	# 		self.direction += 0.1
	# 	if move.startswith('L'):
	# 		self.direction -= 0.1
	# 	if move.startswith('T') and self.fuel > 0:
	# 		self.fuel -= 0.1
	# 		self.acceleration += 1

	# def check_collision(self, frame):
	# 	x, y = map(int, self.position)
	# 	if x >= 0 and x < 1000 and y >= 0 and y < 1000 and frame[x][y] == 255:
	# 		return True
	# 	return False

class Game():
	def __init__(self):
		self.players = []
		self.obstacles = []
		self.holes = []
		self.end = 0
		pygame.init()
		pygame.font.init()
		self.surface = pygame.display.set_mode((SIZE, SIZE))

	def add_player(self, player):
		self.players.append(player)

	def add_obstacle(self, obstacle):
		self.obstacles.append(obstacle)

	def add_hole(self, hole):
		self.holes.append(hole)

	def draw_rect(self, x, y, w, h, val):
		pygame.draw.rect(self.surface, (val, val, val), pygame.Rect(x, y, w, h))

	def finished(self):
		if self.end:
			return True
		return False

	def draw_frame(self, status = 'running'):
		self.draw_rect(0, 0, SIZE, SIZE, 0)
		frame = {
			'obstacles': [],
			'ball': {},
			'hole': {},
			'status': status
		}
		for idx, obstacle in enumerate(self.obstacles):
			if obstacle.x >= 0 and obstacle.x < SIZE and obstacle.y >= 0 and obstacle.y < SIZE:
				self.draw_rect(obstacle.x, obstacle.y, obstacle.w, obstacle.h, 255)
				frame['obstacles'].append({'x': obstacle.x, 'y': obstacle.y, 'w': obstacle.w, 'h': obstacle.h})
		for idx, player in enumerate(self.players):
			x, y = map(int, player.position)
			pygame.draw.circle(self.surface, (100, 100, 100), (x, y), 10)
			frame['ball'] = {'x': x, 'y': y}
		for idx, hole in enumerate(self.holes):
			if hole.x >= 0 and hole.x < SIZE and hole.y >= 0 and hole.y < SIZE:
				pygame.draw.circle(self.surface, (100, 0, 0), (hole.x, hole.y), 10)
				frame['hole'] = {'x': hole.x, 'y': hole.y, 'r': hole.r}
			if hole.check(self.players[0]):
				frame['status'] = 'ok'
				self.end = 1
		pygame.display.flip()
		return json.dumps(frame)