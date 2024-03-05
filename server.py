import socket
from game import Game, Car, Obstacle, Hole
import math
import random
import time
import json

def make_move(conn, game, player):
	conn.send(game.draw_frame('make_move').encode('ascii'))
	try:
		json_val = json.loads(conn.recv(1024).decode('ascii'))
		player.direction = json_val['direction']
		player.acceleration = json_val['acceleration']
	except:
		conn.send('not a valid json'.encode('ascii'))
		return

	for i in range(200):
		if player.velocity == 0 and player.acceleration == 0 or game.finished():
			break
		player.update(game.obstacles)
		conn.send(game.draw_frame().encode('ascii'))
		time.sleep(0.1)


def run():
	sock = socket.socket()
	port = 8342
	sock.bind(('', port))	
	sock.listen(5)

	conn = sock.accept()[0]
	conn.send('x'.encode('ascii'))
	ret = conn.recv(1024).decode('ascii')
	assert ret.startswith('x')
	game = Game()
	player = Car()
	game.add_player(player)
	game.add_hole(Hole(random.randint(0, 999), random.randint(0, 999)))
	for i in range(5):
		game.add_obstacle(Obstacle(random.randint(0, 999), random.randint(0, 999)))		
	while not game.finished():
		make_move(conn, game, player)
	conn.close()
run()
