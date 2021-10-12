import os
import librosa
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm

import plot as pl
import settings as sett
import processing as proc

from functions.Sample import Sound
from matplotlib.widgets import Slider

from scipy import signal


warnings.filterwarnings("ignore")

paths = sett.paths()
params = sett.parameters()
args = sett.arguments().args



if args.generate:
	for file in tqdm(os.listdir(paths.path2Sample)):

		#Extract data
		sample, samplerate = librosa.load(os.path.join(paths.path2Sample, file),
										  sr=None, mono=True, offset=0.0, duration=None)

		tonotopic_maps = np.load(os.path.join(paths.path2Data, 'INT_Sebmice_alignedtohorizon.npy'))
		tonotopic_maps = proc.downscale_tmaps(tonotopic_maps, block_size=(4, 4))
		tonotopic_maps = tonotopic_maps[1:-1, :, :]

		tonotopic_maps = [t[125:375, 200:440] for t in tonotopic_maps] # Reframe tmaps to fit implant size 


		# Remove weighted map at the end and white noise at the beginning


		# Normalization of tonotopic maps and inversion (bright spots should be of interest)
		for i, tmap in enumerate(tonotopic_maps):
			tonotopic_maps[i] = 1 - (tmap - np.min(tmap))/(np.max(tmap) - np.min(tmap))

		# If sample is in stereo take only one track
		if sample.ndim > 1:
			sample = np.transpose(sample[:-len(sample)/2, 0])

		# Compute spectrogram
		specgram, frequencies, times = proc.spectro(sample, samplerate, plot=False)

		# Compute rectangle stimulation if needed
		if args.rectangle is not None:
			rect_stim, weighted_tmap, min_4, min_32 = proc.rectangle_stim(tonotopic_maps[0], tonotopic_maps[2], args.rectangle)
			#frequencies = np.linspace(4000, 32000, num=801)
			magnitudes = proc.rectangle_windowing(specgram, frequencies, n_rectangle=args.rectangle)


			all_maps = []
			for magnitude in magnitudes:
				buffer_map = np.copy(weighted_tmap)
				for rect, mag in zip(rect_stim, magnitude):
					buffer_map[rect[:, 0], rect[:, 1]] = mag
				all_maps.append(buffer_map)
			all_maps = np.array(all_maps)
			all_maps = all_maps[:400]

			if args.plot:
				pl.rectangle_mp4(min_4, min_32, all_maps)
			single_map = True

		else:
			magnitudes = proc.square_windowing(specgram, frequencies)
			all_maps = np.array([tonotopic_maps * mag[:, np.newaxis, np.newaxis] for mag in magnitudes])
			single_map = False

		projections = proc.implant_projection(all_maps, single_map=single_map)

		np.save(os.path.join(paths.path2Output, file[:-4] + '_rectangles_6_40'), projections)
		#pl.cube_show_slider(np.swapaxes(projections[1:,:,:], 0, 2))


# Do a different processinf for AM sound
# Revoir le gaussian windowing qui est chelou. Fréquence pas stable.

def correlate():
	matrix_to_corr = []
	for file in np.sort(os.listdir(paths.path2Output)):
		if file.endswith('.npy'):
			curr_proj = np.load(os.path.join(paths.path2Output, file))
			curr_proj = np.mean(curr_proj, axis=0)
			print(curr_proj.shape)
			matrix_to_corr.append(curr_proj)
	matrix_to_corr = np.array(matrix_to_corr)
	matrix_to_corr = np.array([np.matrix.flatten(proj) for proj in matrix_to_corr])
	print(matrix_to_corr)
	correlation_matrix = np.corrcoef(matrix_to_corr, matrix_to_corr)
	plt.imshow(correlation_matrix)
	plt.show()




	# # in dev
	# a = np.load(os.path.join(paths.path2Output,'PT_16000Hz_500ms_70dB_rectangles_6_40.npy'))
	# b = np.load(os.path.join(paths.path2Output,'PT_6000Hz_500ms_70dB_rectangles_6_40.npy'))

	# fig, axs = plt.subplots(3)
	# axs[0].imshow(a[0])
	# axs[1].imshow(b[0])
	# axs[2].imshow(signal.correlate(a[0], b[0]))
	# plt.show()


	# c = proc.correlate_representations(a, b)
	# print(c.shape)

	# pl.cube_show_slider(np.swapaxes(c, 0, 2))
correlate()
