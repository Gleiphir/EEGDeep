import time
import pygame
from cortex import Cortex

from cortex import SUB_REQUEST_ID

import json
import random
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

from threading import Thread



WINDOW_WIDTH = 30

FRAME_WIDTH = 800
FRAME_HEIGHT = 800

radian = np.pi * 2 / 360.0
xx, yy = np.mgrid[:FRAME_WIDTH, :FRAME_HEIGHT]

class CortexListener(Cortex,Thread):
	def __init__(self, user, debug_mode=False):
		Cortex.__init__(self, user, debug_mode=debug_mode)
		Thread.__init__(self)
		self.stream = ['eeg']
		self.rolling = np.zeros([WINDOW_WIDTH,14])
		self.P8 = []
		self.val = 0.0



	def update_map(self,new:list,alpha):
		#PARA_R, PARA_G, PARA_B = new[0],  new[1], new[2]
		#PARA_R, PARA_G, PARA_B = abs(new[0]-new[1]), abs(new[1]-new[2]), abs(new[0]-new[2])
		npl = np.array(new)
		norm_npl = np.abs(( npl - np.mean(npl) )/ np.std(npl) )
		PARA_R, PARA_G, PARA_B = norm_npl[0], norm_npl[1], norm_npl[2]#np.std(npl)
		r = (np.sin(xx * radian * PARA_R) * 0.5 + 0.5 ) * 255#PARA_R
		g = (np.sin(yy * radian * PARA_G) * 0.5 + 0.5 ) * 255#PARA_G
		b = r * g % 256#PARA_B
		#b = np.abs(np.sin(xx * yy * radian * 0.025)) * PARA_B
		image = np.stack([r , g , b ], axis=2)
		#image = image
		image = image * alpha
		# print(image.shape)
		print("{:.2f},{:.2f},{:.2f},{:.2f}".format(PARA_R, PARA_G,PARA_B,alpha))
		self.rolling = image.astype(np.uint8)

	def get_map(self):
		return self.rolling

	def getP8(self):
		return self.val

	def run(self):

		print('subscribe request --------------------------------')
		sub_request_json = {
			"jsonrpc": "2.0",
			"method": "subscribe",
			"params": {
				"cortexToken": self.auth,
				"session": self.session_id,
				"streams": self.stream
			},
			"id": SUB_REQUEST_ID
		}

		self.ws.send(json.dumps(sub_request_json))
		new_data = self.ws.recv()
		print(json.loads(new_data))
		while True:
			new_data = self.ws.recv()
			#print(json.loads(new_data))
			data = json.loads(new_data)["eeg"]
			print(data)
			#self.val = data[8]
			"""
			self.P8.append(data[8])
			if len(self.P8) > 100:
				self.P8 = self.P8[-100:]
			lightness = 0.3 + 0.7* np.sin(np.abs( np.mean(np.array(self.P8[-10:])) - np.mean(np.array(self.P8)) ) / np.std(np.array(self.P8)))
			self.update_map(new_batch,lightness)
			print(new_data)
			"""





"""
class Subcribe():
	def __init__(self):
		self.c = Cortex(user, debug_mode=True)
		self.c.do_prepare_steps()

	def sub(self, streams):
		self.c.sub_request(streams)

	def showMap(self):
		return 
"""
# -----------------------------------------------------------
# 
# SETTING
# 	- replace your license, client_id, client_secret to user dic
# 	- specify infor for record and export
# 	- connect your headset with dongle or bluetooth, you should saw headset on EmotivApp
#
# 
# RESULT
# 	- subcribed data type should print out to console log
# 	- "cols" contain order and column name of output data
# 
# 
#	{"id":6,"jsonrpc":"2.0","result":{"failure":[],"success":[{"cols":["COUNTER","INTERPOLATED","T7","T8","RAW_CQ","MARKER_HARDWARE","MARKERS"],"sid":"0fd1c571-f0ec-4aa0-bb71-b4fa2b9c7504","streamName":"eeg"}]}}
# 	{"eeg":[4,0,4222.476,4202.952,0.0,0,[]],"sid":"866e47d8-d7e6-4cfa-87b7-c4f956d6c429","time":1590984953.6683}
# 	{"eeg":[5,0,4220.571,4204.857,0.0,0,[]],"sid":"866e47d8-d7e6-4cfa-87b7-c4f956d6c429","time":1590984953.6761}
# 	{"eeg":[6,0,4219.143,4207.238,0.0,0,[]],"sid":"866e47d8-d7e6-4cfa-87b7-c4f956d6c429","time":1590984953.6839}
# 	{"eeg":[7,0,4218.667,4198.667,0.0,0,[]],"sid":"866e47d8-d7e6-4cfa-87b7-c4f956d6c429","time":1590984953.6917}
# -----------------------------------------------------------

user_zh = {
	"license" : "92463224-a21b-4a3c-a161-cdfa668bc836",
	"client_id" : "55Vq8DgXpteDXJ1PHobUOkgFf0PLvGoBjEoXo78V",
	"client_secret" : "PbKQmZ9XVCeEd9e4hrLLvXWwVbZKJOa1dEaDdJpaPm3H6gRel120Th8TRpg9BvWnXD3WdrGu6FipKdIQ9scxT4HSeIL9XU3pqRz6qAt3QavkX3jUtg7isVxqpJMDobtF",
    "debit" : 100
}


#s = Subcribe()
if __name__ == '__main__':
	listener = CortexListener(user_zh,debug_mode=True)

	# sub multiple streams
	#streams = ['eeg','mot','met','pow']
	listener.do_prepare_steps()
	listener.start()

