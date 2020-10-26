import os 
import librosa
import warnings
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

import plot as pl 
import settings as sett
import processing as proc

from functions.Sample import Sound

warnings.filterwarnings("ignore")

paths = sett.paths()
params = sett.parameters()
args = sett.arguments().args

# Complex sound
modul = Sound()
modul.freq_modulation(4000, 16000,duration=500)

am = Sound()
am.amplitude_modulation(16000, 15, duration=1500)

harm = Sound()
harm.multi_freqs([5000, 7000, 15000, 30000], duration=1500)

pure = Sound()
pure.simple_freq(8000, duration=500)

final = modul + am +harm + pure
final.save_wav(name='final_test', path='../Samples')




#Extract data
sample, samplerate = librosa.load(os.path.join(paths.path2Sample, 'final_test.wav'),
								  sr=None, mono=True, offset=0.0, duration=None)

tonotopic_maps = np.load(os.path.join(paths.path2Data, 'INT_Sebmice_alignedtohorizon.npy'))
tonotopic_maps = proc.downscale_tmaps(tonotopic_maps, block_size=(4, 4))

# Remove weighted map at the end and white noise at the beginning
tonotopic_maps = tonotopic_maps[1:-1, :, :]

# Normalization of tonotopic maps and inversion (bright spots should be of interest)
for i, tmap in enumerate(tonotopic_maps):
	tonotopic_maps[i] = 1 - (tmap - np.min(tmap))/(np.max(tmap) - np.min(tmap))

# If sample is in stereo take only one track
if sample.ndim > 1:
	sample = np.transpose(sample[:-len(sample)/2, 0])

# Compute spectrogram
specgram, frequencies, times = proc.spectro(sample, samplerate)

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
	magnitudes = proc.gaussian_windowing(specgram, frequencies)
	all_maps = np.array([tonotopic_maps * mag[:, np.newaxis, np.newaxis] for mag in magnitudes])
	single_map = False


projections = proc.implant_projection(all_maps, single_map=single_map)
