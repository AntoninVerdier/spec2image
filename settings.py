# Hearing peaks at 16 kHz in mice  : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2949429/

import os
import numpy as np 

class paths():
	def __init__(self):
		self.path2Data = '../Data/'
		self.path2Sample = '../Samples/'
		self.path2Output = '../Output/'

class parameters():
	def __init__(self):
		self.size_implant = 40
		self.freq_resolution = 4
		self.freqs = [4e3, 16e3, 32e3]


# Code review
# Add amplitude
# bruit gaussien
# log spectrogram
# humain x4
# seuile carte tonotopic
# soustraction carte tonotopic
