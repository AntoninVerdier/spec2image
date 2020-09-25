import os
import librosa
import numpy as np

from librosa import display 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import settings as sett 
paths = sett.paths()

def waveplot(sample, samplerate):
	""" Create a waveplot figure of the raw sample
	"""
	plt.figure(figsize=(12, 6))

	display.waveplot(y=sample, sr=samplerate)
	
	plt.title('Waveplot of sound sample')
	plt.xlabel('Time (sec)')
	plt.ylabel('Amplitude')
	
	plt.savefig(os.path.join(paths.path2Output, 'sample_waveplot.png'))
	plt.close()


def fft(sample, samplerate, fft):
	n = len(sample)
	T = 1/samplerate
	xf = np.linspace(0, 1/(2*T), n//2)
	yf = 2/n * np.abs(fft[:n//2])
	
	plt.plot(xf, yf)
	
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Magnitude')
	
	plt.savefig(os.path.join(paths.path2Output, 'sample_fft.png'))
	plt.close()

def spectrogram(spectrogram, time=None, frequencies=None):
	""" Create a simple spectrogram of the sound sample
	"""
	plt.imshow(spectrogram, interpolation=None)

	plt.title('Spectrogram of sound sample')
	plt.xlabel('Time (sec)')
	plt.ylabel('Frequency (Hz)')
	plt.xticks(time)
	plt.yticks(frequencies)

	plt.savefig(os.path.join(paths.path2Output, 'sample_spectrogram.png'))
	plt.close()

def gif_projections(tmaps):
	fig, axs = plt.subplots(1, 3)

	ims = []
	for t in range(tmaps.shape[0]):
		im0 = axs[0].imshow(tmaps[t, 0, :, :], cmap='gray', vmin=0, vmax=1)
		im1 = axs[1].imshow(tmaps[t, 1, :, :], cmap='gray', vmin=0, vmax=1)
		im2 = axs[2].imshow(tmaps[t, 2, :, :], cmap='gray', vmin=0, vmax=1)
		#im3 = axs[1, 1].imshow(tmaps[t, 3, :, :], cmap='gray')
		ims.append([im0, im1, im2])

	ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
	                                repeat_delay=1000)

	ani.save(os.path.join(paths.path2Output, 'animation.gif'), writer='imagemagick', fps=30)

	# fig, axs = plt.subplots(2, 2)

	# ims = []
	# for t in range(tmaps.shape[0]):
	# 	axs[0, 0].imshow(tmaps[t, 0, :, :], cmap='gray')
	# 	axs[0, 1].imshow(tmaps[t, 1, :, :], cmap='gray')
	# 	axs[1, 0].imshow(tmaps[t, 2, :, :], cmap='gray')
	# 	axs[1, 1].imshow(tmaps[t, 3, :, :], cmap='gray')
	# 	ims.append([axs])

	# ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
	#                                 repeat_delay=1000)

	# ani.save('animation.gif', writer='imagemagick')
	# plt.show()

	# fig = plt.figure()

	# ims = []
	# for image in gen:
	#     im = plt.imshow(image[0, :, :], animated=True, cmap="Greys")
	#     plt.axis("off")
	#     ims.append([im])

	# ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
	#                                 repeat_delay=1000)
# # print(tonotopic_maps)
# print(tonotopic_maps.shape)


# for i in range(5):
# 	print(np.max(tonotopic_maps[i]))
# 	plt.imshow(tonotopic_maps[0, :, :], cmap='gray')
# 	plt.show()
# 	plt.close()





