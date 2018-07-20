"""
Collection of functions used to perform feature 
extraction for speaker localization. Includes filterbanks, 
enery detection, framing/segmentation, etc. 

"""


# Python implementation of gammatone based on the MATLAB code by:  Christopher Hummersone
# Link to original source: http://www.mathworks.com/matlabcentral/fileexchange/32212-gammatone-filterbank
#
# Navid Shokouhi - January 2016 

import numpy as np
import scipy.signal
import scipy.io.wavfile as wav
import pylab 
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def rectify(x):
	"""
	returns 0 for negative entries of x. 
	"""
	return np.array([max(i,0) for i in x])

def delay_signal(x,delay=100):
	"""
	delay x by delay samples
	"""
	x = np.roll(x,delay)
	x[:delay] = 0
	return x

def moving_average(x,M=10):
	"""
		Calculate order M moving average of signal x. 
		Inputs: 
			x: inputs signal. 
			M: order of moving average filter
		Outputs:
			returns filtered signal.  
	"""
	return np.convolve(x, np.ones((M,))/(1.0*M), mode='valid')


def cochlear_model_processing(x,fs,time_avg=0.03):
	"""
	Needs comments
	"""
	x = rectify(x)
	x_avg = np.zeros(x.shape)
	temp = moving_average(x,int(fs*time_avg)+1)
	x_avg[-max(temp.shape):] = temp
	x_avg = delay_signal(x_avg)
	x_modulated = rectify(x - x_avg)
	x_threshold = rectify(x - 2*x_avg) # 6dB threshold
	above_threshold = 1.0*(x_threshold>0)
	onset_offsets = np.diff(above_threshold)
	t_onset = np.where(onset_offsets==1)[0]
	t_offset = np.where(onset_offsets==-1)[0]
	pulses = 0*x
	pulse_length = int(fs*20*1e-3) # 20 μsec pulse
	for i in range(max(t_onset.shape)-1):
		if (t_onset[i]<t_offset[i]):
			h = sum(np.sqrt(x_threshold[t_onset[i]:t_offset[i]]))
		else: 
			h = 0
		pulse_center = int((t_onset[i]+t_offset[i])/2)
		pulses[pulse_center-int(pulse_length/2):pulse_center+int(pulse_length/2)] = h
	return pulses


def apply_fbank(x,fs,cfs,align=False,hilbert_envelope=False):
    """
        x:        input signal (numpy array)
        fs:       sampling rate (integer)
        cfs:      center frequencies (numpy array)
        align:    allow phase alignment across filters. 
    """
    n_channels = len(cfs)
    filterOrder = 4
    gL = int(0.128*fs) # minimum filter length is 128 msec. This must be a power of 2. 
    b = 1.019*24.7*(4.37*(cfs/1000)+1); # rate of decay or bandwidth
    gt = np.zeros((gL,n_channels));  # Initialise IR
    tc = np.zeros(cfs.shape);  # Initialise time lead
    phase = 0

    tpt = (2*np.pi)/fs
    gain = ((1.019*b*tpt)**filterOrder)/6; # based on integral of impulse

    tmp_t = np.array([range(gL)])*1.0/fs

    # calculate impulse responses:
    y = np.zeros((len(x),n_channels))
    for i in range(n_channels):
        gain_term = gain[i]*fs**3
        poly_term = tmp_t**(filterOrder-1)
        damp_term = np.exp(-2*np.pi*b[i]*tmp_t)
        oscil_term = np.cos(2*np.pi*cfs[i]*tmp_t+phase)
        gt[:,i] = gain_term*poly_term*damp_term*oscil_term; 
        bm = scipy.signal.fftconvolve(x,gt[:,i].reshape((gL,1)))
        y[:,i] = bm[:len(x),0]

        if hilbert_envelope:
            bm_hilbert = scipy.signal.hilbert(bm[:len(x),0])
            y[:,i] = abs(bm_hilbert)

    return y, b

def ErbRateToHz(erb):
    return (10**(erb/21.4)-1)/4.37e-3

def HzToErbRate(hz):
    return (21.4*np.log10(4.37e-3*hz+1))

def make_centerFreq(minCf,maxCf,n_channels):
    return ErbRateToHz(np.linspace(HzToErbRate(minCf),HzToErbRate(maxCf),n_channels));

def check_filterbank():
	"""
	Example on how to extract gamma-tone filter outputs.
	"""
	sample_file = "../test_dir/CMU_ARCTIC/cmu_us_bdl_arctic/wav/arctic_a0001.wav"
	import os
	if not(os.path.isfile(sample_file)):
		raise("sample file does not exist. Fix path.")
	
	rate,sig = wav.read(sample_file)
	x = sig.reshape((len(sig),1))
	fs = rate
	cfs = make_centerFreq(20,3800,40)
	hilbert_x,bandwidths = apply_fbank(x,fs,cfs,hilbert_envelope=True)
	filtered_x,_ = apply_fbank(x,fs,cfs,hilbert_envelope=False)
	for i in range(40):
		x_i = filtered_x[:,i]/abs(max(filtered_x[:,i]))
		h_i = hilbert_x[:,i]/abs(max(hilbert_x[:,i]))
		z_1 = cochlear_model_processing(x_i,fs)
		pylab.plot(x_i + bandwidths[i],color='b')
		pylab.plot(z_1 + bandwidths[i],color='r')
		pylab.plot(z_2 + bandwidths[i],color='g')
		pylab.ylabel('bandwidths (linearly related to center freqs.)')
		pylab.xlabel('time (samples)')
	pylab.show()



if __name__=='__main__':
	sample_file = "../test_dir/CMU_ARCTIC/cmu_us_bdl_arctic/wav/arctic_a0001.wav"
	import os
	if not(os.path.isfile(sample_file)):
		raise("sample file does not exist. Fix path.")
	
	rate,sig = wav.read(sample_file)
	x = sig.reshape((len(sig),1))
	fs = rate
	n_channels = 40
	cfs = make_centerFreq(20,3800,n_channels)
	print(x.shape)
	x_filtered,_ = apply_fbank(x,fs,cfs,align=False,hilbert_envelope=False)
	x_neural = np.zeros(x_filtered.shape)
	for i in range(n_channels):
		x_neural[:,i] = cochlear_model_processing(x_filtered[:,i],fs)
		pylab.plot(x_neural[:,i]+cfs[i])
	pylab.show()




	