# Hearing peaks at 16 kHz in mice  : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2949429/

import os
import argparse
import numpy as np

import matplotlib.pyplot as plt

class paths():
	def __init__(self):
		self.path2Data = '../Data/'
		self.path2Sample = 'Samples/'
		self.path2Output = '../Output/'

class parameters():
	def __init__(self):
		self.size_implant = 200
		self.freq_resolution = 4
		self.freqs = [4e3, 16e3, 32e3]

		plt.rcParams['figure.figsize'] = [20, 12]
		plt.rcParams['figure.dpi']

class arguments():
	def __init__(self):
		parser = argparse.ArgumentParser(description='Parameters for computing')
		parser.add_argument('--rectangle', '-r', type=int, default=None,
						    help='Number of stimulation rectangles')
		parser.add_argument('--plot', '-p', action='store_true',
						    help='If present, stimulations will be plotted')
		parser.add_argument('--generate', '-g', action='store_true',
						    help='If present, stimulations will be generated')		
		self.args = parser.parse_args()


# Code review
# Add amplitude
# bruit gaussien
# log spectrogram
# humain x4
# seuile carte tonotopic
# soustraction carte tonotopic
