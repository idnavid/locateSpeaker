from __future__ import print_function
import time
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def simulate_data(n_sources,n_micarrays,PLOT_ROOM=False):
	"""
	Using pyroomacoustics to simulate
	sound propagation in a shoebox room and record the result
	to a wav file.
	Room dimensions are 6 x 6 x 2.4

	Inputs: 
		n_sources: number of sources (i.e., speakers)
		n_micarrays: number of microphone arrays (= number_of_microphones/2)
		PLOT_ROOM: plot simulated room. 
				   there's currently a bug in the plots. I keep getting mirror
				   images of the sources outside the room. 

	Outputs:
		array_recordings: numpy array containing all of the recorded signals. 
						  [n_micarrays x 2 x number_of_samples]
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
		# position = np.array([np.random.randint(1,6),np.random.randint(1,6),np.random.uniform(1.8,2.1)]).T
		position = np.array([5,5,2]).T
		source_positions.append(position)
		room.add_source(position, delay=(10.0)*i,signal=source_signals[i])
	
	# simulate recordings for Microphone Arrays one at a time, 
	# each consists of a linear array of 2 microphones 0.1m apart.  
	array_recordings = []
	mic_positions = []
	for m in range(n_micarrays):
		# x1 = np.random.uniform(0.2,5.8)
		# y1 = np.random.uniform(0.2,5.8)
		x1 = 0.1
		y1 = 0.1
		z1 = 1.0 # array is always a 1.0 m high
		x2 = x1
		y2 = y1 + 0.1 # length of array is 0.1
		z2 = z1 
		R = np.array([[x1, x2], [y1, y2], [z1,  z2]])  # [[x], [y], [z]]
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
	temp,fs,_,_ = simulate_data(1,4,PLOT_ROOM=False)
