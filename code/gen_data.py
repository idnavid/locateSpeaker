from __future__ import print_function
import time
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def simulate_data(n_sources,n_micarrays,fixed_position=False,PLOT_ROOM=False):
	"""
	Using pyroomacoustics to simulate
	sound propagation in a shoebox room and record the result
	to a wav file.
	All of the microphone arrays consists of only two mics (left and right). 
	Room dimensions are 6 x 6 x 2.4

	Inputs: 
		n_sources:      number of sources (i.e., speakers)
		n_micarrays:    number of microphone arrays (= number_of_microphones/2)
		fixed_position: use a set of predetermined positions for the sources/microphones
						If false, the simulater uses random positions. (default False)
		PLOT_ROOM:      plot simulated room. 
				        there's currently a bug in the plots. I keep getting mirror
				        images of the sources outside the room. (default False)

	Outputs:
		array_recordings: numpy array containing all of the recorded signals. 
						  [n_micarrays x 2 x number_of_samples]
		fs: 			  sampling rate (Hz)
		mic_positions:    numpy array containing microphone positions in xyz coordinates.
						  [n_micarrays x 2 x 3]
		source_positions: numpy array containing source positions in xyz coordinates.
					      [n_sources x 3]
	"""

	assert(n_sources>0 and n_sources<10)
	assert(n_micarrays>0)

	source_signals = []
	source_positions = []
	for i in range(n_sources):
		fs,sig = wavfile.read('../test_dir/CMU_ARCTIC/cmu_us_bdl_arctic/wav/arctic_a000'+str(i+1)+'.wav')
		source_signals.append(sig)

	corners = np.array([[0,0], [0,6], [6,6], [6,0]]).T 
	room = pra.Room.from_corners(corners)
	room.extrude(2.4)

	# Add Sources
	for i in range(n_sources):
		if not(fixed_position):
			theta = np.random.uniform(0,2*np.pi)
			r = np.random.uniform(0,2)
			x = 3 + r*np.cos(theta)
			y = 3 + r*np.sin(theta)
			z = np.random.uniform(1.8,2.1)
			position = np.array([x,y,z])

		else:
			# NOTE:
			#   This scenario only works for a single position. 
			#   I need to write a function for up to 10 positions. 
			position = np.array([5,5,2]).T

		source_positions.append(position)
		room.add_source(position, delay=(10.0)*i,signal=source_signals[i])
	
	# simulate recordings for Microphone Arrays one at a time, 
	# each consists of a linear array of 2 microphones 0.1m apart.  
	array_recordings = []
	mic_positions = []
	for m in range(n_micarrays):
		# NOTE: 
		#   I wasn't able to get pyroomacoustics to simulate
		#   all microphone arrays together. So, I had to 
		#   import them one by one and generate their responses.
		if not(fixed_position):
			theta = np.random.uniform(0,2*np.pi)
			r = np.random.uniform(2.5,2.9)
			x1 = 3 + r*np.cos(theta)
			y1 = 3 + r*np.sin(theta)

		else:
			# NOTE:
			#   This scenario only works for a single position. 
			#   I need to write a function for up to 10 positions. 
			x1 = 0.1
			y1 = 0.1

		z1 = 1.0 # array is always a 1.0 m high
		x2 = x1
		y2 = y1 + 0.1 # length of array is 0.1
		z2 = z1 
		R = np.array([[x1, x2], [y1, y2], [z1,  z2]]) 
		mic_positions.append(R)
		room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
		
		room.simulate()
		array_recordings.append(room.mic_array.signals)

		if PLOT_ROOM:
			fig, ax = room.plot()
			ax.set_xlim([0.0, 6.0])
			ax.set_ylim([0.0, 6.0])
			ax.set_zlim([0.0, 2.4])
			plt.show()
			room.plot_rir()
			plt.show()
	return np.array(array_recordings),fs,np.array(mic_positions),np.array(source_positions)

if __name__=='__main__':
	temp,fs,mic_array,source_array = simulate_data(1,12,fixed_position=False,PLOT_ROOM=True)
	print(mic_array.shape,source_array.shape)
