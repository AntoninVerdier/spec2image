# spec2image
This algorithm converts arbitrary sound files into a 2D-shape sequence of images, consistant with the tonotopic maps of the auditory cortex of the mouse. 

## Pre-processing
A Fourier transform is performed on the raw sound signal every `window * sample_rate` point. Default parameters are consistant with a speech recognition task `window = 20ms`, `overlap = 50%`. A spectrogram is then generated.

For memory purposes, tonotopic maps of the auditory cortex are first downscaled. Each tonotopic map has a original shape of `'2000, 2500`. Averaging is done using blocks of shape `(4, 4)`, bringing final shape to `(500, 625)`.

## Weighting of tonotopic maps
For each timepoint in the spectrogram, a vector encapsulating all frequencies magnitudes is extracted. Followinf the tonotopic map selectivity (currently 4kHz, 16kHz and 32kHz) magnitudes around each map frequency are summed (frequencies of interest in the signal are selected using a gaussian window centered on map selectivity). 

Each tonotpic map is then multiplied at any given time by the corresponding magnitude. A animated gif can be generated for visualisation.



