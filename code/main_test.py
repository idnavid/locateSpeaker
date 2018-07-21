
from gen_data import *
from feature_extraction import *

x_mic,fs,mic_positions,source_positions = simulate_data(1,1)
n_arrays,_,n_samples = x_mic.shape

c = 343.0 # speed of sound m/s
for i in range(source_positions.shape[0]):
	p_mic1 =  mic_positions[0,:,0]
	p_mic2 =  mic_positions[0,:,1]
	p = source_positions[i,:]
	d1 = np.linalg.norm(p-p_mic1)
	d2 = np.linalg.norm(p-p_mic2)
	tau_true = np.abs(d1 - d2)/c
print(tau_true)

n_channels = 16
cfs = make_centerFreq(20,3800,n_channels)
tau = np.zeros((n_channels,n_arrays))
for i in range(n_arrays):
	x1 = x_mic[i,0,:].reshape(n_samples,1)
	x2 = x_mic[i,1,:].reshape(n_samples,1)

	x1_filtered,_ = apply_fbank(x1,fs,cfs)
	x2_filtered,_ = apply_fbank(x2,fs,cfs)
	x1_neural = np.zeros(x1_filtered.shape)
	x2_neural = np.zeros(x2_filtered.shape)
	for j in range(n_channels):
		x1_neural[:,j] = cochlear_model_processing(x1_filtered[:,j],fs,0.03,0.001)
		x2_neural[:,j] = cochlear_model_processing(x2_filtered[:,j],fs)
		temp = estimate_tdoa(x1_neural[:,j],x2_neural[:,j],int(0.1*fs),100,fs)
		print(temp.T)
print(tau)