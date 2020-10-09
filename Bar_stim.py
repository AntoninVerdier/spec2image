#!/usr/bin/env python
# coding: utf-8

import os 
import math
import scipy
import numpy as np 
import matplotlib.pyplot as plt 

import settings as sett
from functions.Sample import Sound

from tqdm import tqdm
import time

plt.rcParams['figure.figsize'] = [20, 12]
plt.rcParams['figure.dpi'] 


tonotopic_maps = np.load(os.path.join('../Data/INT_Sebmice_alignedtohorizon.npy'))

tmap4, tmap32 = tonotopic_maps[1], tonotopic_maps[3]

def rectangle_stim(tmap4, tmap32, n_rectangles, width_rect=0.4, squared=False):
	tmap4 = 1 - (tmap4 - np.min(tmap4)) / (np.max(tmap4) - np.min(tmap4))
	tmap32 = 1 - (tmap32 - np.min(tmap32)) / (np.max(tmap32) - np.min(tmap32))

	# Get minima
	min_1 = np.unravel_index(tmap4.argmax(), tmap4.shape)
	min_2 = np.unravel_index(tmap32.argmax(), tmap32.shape)

	if squared:
		tmap4, tmap32 = tmap4**2, tmap32**2


	weighted_tmap = tmap32 - tmap4

	distance_min = math.sqrt((min_1[0] - min_2[0])**2 + (min_1[1] - min_2[1])**2)
	
	vector = np.array([min_2[0] - min_1[0], min_2[1] - min_1[1]])
	v_alpha = vector * 1 / n_rectangles
	v_theta = np.array([vector[1], - vector[0]]) * width_rect

	rect_stim = []
	for i in range(n_rectangles):
		origin = np.array([min_1[0], min_1[1]]) + i * v_alpha

		rect = np.array([[origin[0] - v_theta[0], origin[1] - v_theta[1]],
						 [origin[0] - v_theta[0] + v_alpha[0], origin[1] - v_theta[1] + v_alpha[1]],
						 [origin[0] + v_theta[0] + v_alpha[0], origin[1] + v_theta[1] + v_alpha[1]],
						 [origin[0] + v_theta[0], origin[1] + v_theta[1]]])

		
		idx_x = np.arange(int(np.min(rect[:, 0])), int(np.max(rect[:, 0])))
		idx_y = np.arange(int(np.min(rect[:, 1])), int(np.max(rect[:, 1])))


		edges = np.array([[rect[j-1], rect[j]] for j in range(4)])

		inside = []
		for idx in idx_x:
			for idy in idx_y:
				score = 0
				for edge in edges:
					D = (edge[1, 0] - edge[0, 0]) * (idy - edge[0, 1]) - (idx - edge[0, 0]) * (edge[1, 1] - edge[0, 1])
					if D < 0:
						score += 1
				if score == 4:
					inside.append([int(idx), int(idy)])
		print(inside)
		rect_stim.append(np.array(inside))
	
	return rect_stim, weighted_tmap, min_1, min_2

rect_stim, weighted_tmap, min_4, min_32 = rectangle_stim(tmap4, tmap32, 5)

for i, rect in enumerate(rect_stim):
	weighted_tmap[rect[:, 1], rect[:, 0]] = 0.2 * i



# Script for checking
#plt.scatter(rect[:, 0], rect[:, 1])
plt.scatter(min_4[0], min_4[1], marker='o', c='red')
plt.scatter(min_32[0], min_32[1], marker='o', c='red')
plt.plot([min_4[0], min_32[0]], [min_4[1], min_32[1]])
plt.imshow(weighted_tmap, cmap='coolwarm')

plt.show()





