# Hearing peaks at 16 kHz in mice  : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2949429/

import os
import numpy as np 

class paths():
	def __init__(self):
		self.path2data = '../Samples/'
		self.path2Output = '../Output/'

class parameters():
	def __init__(self):
		self.size_implant = 10
		self.freq_resolution = 20
		self.freqs = np.logspace(12e3, 20e3, num=self.freq_resolution)
