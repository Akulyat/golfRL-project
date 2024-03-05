import json
import socket
import math

def run():
	sock = socket.socket()
	port = 8342
	sock.connect(('', port))	
	sock.recv(1024).decode('ascii')
	sock.send('x'.encode('ascii'))
	while True:
		json_val = json.loads(sock.recv(1024).decode('ascii'))
		if json_val['status'] == 'ok':
			break
		if json_val['status'] != 'make_move':
			continue
		A = json_val['ball']
		B = json_val['hole']
		dx = B['x'] - A['x']
		dy = B['y'] - A['y']
		json_response = {
		    'direction': math.atan2(dy, dx),
		    'acceleration': math.sqrt(dx ** 2 + dy ** 2) / 100
		}
		sock.send(json.dumps(json_response).encode('ascii'))
	print('WIN')


run()