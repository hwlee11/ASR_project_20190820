import numpy as np
import math
import cmath
import scipy.io as sio
import scipy.io.wavfile

import librosa

import matplotlib.pyplot as plt
import sys
from scipy.fftpack import dct
np.set_printoptions(threshold=sys.maxsize)

#import sounddevice as sd

class pre_processing:

    def mel_scale(self,f):
        m = 1125*np.log(1+f/700)
        return m

    def inverse_mel(self,m):
        f = 700 * ( np.exp(m/1125) - 1)
        return f

    def pre_emphasise(self,data,k=0.97):

        '''
        s'_n = s_n - k * s_n-1
        '''

        s = []
        s.append(data[0])
        for i in range(1,len(data)):
            s.append(data[i] - k*data[i-1])
        pre_emphasise_data = np.array(s)
        return pre_emphasise_data


    def hamming_window(self,data):

        '''
        cosine-sum window for K = 1 : s'_n = { a - b * cos( 2pi(n-1) / N-1 ) }*s_n 
        hann window a = 0.5, b = 0.5
        hamming window a = 0.54, b = 0.46
        '''

        s = []
        s.append(data[0])
        times = len(data)
        for i in range(1,times):
            s.append( (0.54-0.46*math.cos((2*math.pi*(i-1))/(times-1)))*data[i])
            #s.append( (0.54-0.46*math.cos((2*math.pi*(i))/(times-1)))*data[i])
        hamming_data = np.array(s)
        return hamming_data

    def frame_split(self,data,samplerate,window_dtime=25e-3,frame_dtime=10e-3):
        #
        frames = []
        dt = 1/samplerate
        window_size = math.ceil(window_dtime/dt)
        frame_size = math.ceil(frame_dtime/dt)
        data_size = len(data)
        num_of_frames = math.ceil(((data_size-window_size)/frame_size)+1)
        offset = 0
        for i in range(num_of_frames):
            offset = i*frame_size
            frames.append(np.array(data[offset:offset+window_size]))

        return frames

    def discrete_fourier_transform(self,data,dft_window_size=512):
        N = len(data)
        if N > dft_window_size:
            data = data[:dft_window_size]
        else :
            pad_width = (dft_window_size - N)/2
            data = np.pad(data,(int(np.trunc(pad_width)),int(np.ceil(pad_width))),'constant')
        c = np.zeros(dft_window_size//2+1,complex)
        N = len(data)
        for k in range(dft_window_size//2+1):
            for n in range(N):
                c[k] += data[n]*cmath.exp(-2j*cmath.pi*k*n/N)
        return c

    def full_discrete_fourier_transform(self,data,dft_window_size=512):
        N = len(data)
        if N > dft_window_size:
            data = data[:dft_window_size]
        else :
            pad_width = (dft_window_size - N)/2
            data = np.pad(data,(int(np.trunc(pad_width)),int(np.ceil(pad_width))),'constant')
        c = np.zeros(512,complex)
        N = len(data)
        for k in range(512):
            for n in range(N):
                c[k] += data[n]*cmath.exp(-2j*cmath.pi*k*n/N)
        return c
    def inverse_discrete_fourier_transform(self,data,fff,window_size=512):
        N = len(data)
        y = np.zeros(window_size,complex)
        c_ = data[:N-1].conjugate()
        c_ = c_[::-1]
        data = np.hstack([data,c_[:]])
        N = len(data)-1
        data = data[:N]
        print("비교",data == fff,len(data),len(fff))
        #print('conjugate',len(data),data[:N])
        for n in range(window_size):
            for k in range(N):
                y[n] += data[k]*cmath.exp(2j*cmath.pi*k*n/N)
            y[n] /= window_size
        return y

    def triangular_transform(self,frame,f,num_of_filter=40,fft_window=512):

        #check_list = np.zeros([fft_window])
        filter_bank_frame = np.zeros([num_of_filter])
        fbank = 0
        for m in range(1,num_of_filter+1):
            for k in range( int(f[m-1]),int(f[m+1])):
                if f[m-1] <= k < f[m]:
                    h_k = (k - f[m-1]) / (f[m] - f[m-1]) 
                    fbank += h_k * frame[k]
                    #check_list[k] = h_k
                elif f[m] <= k <= f[m+1]:
                    h_k = (f[m+1]-k) / (f[m+1]-f[m])
                    fbank += h_k * frame[k]
                    #check_list[k] = h_k
            #print(len(check_list),check_list)
            #check_list = np.zeros([fft_window])
            
            filter_bank_frame[m-1] = fbank
            fbank = 0
        #print('mine')
        
        return filter_bank_frame
        

    def filter_bank(self,data,samplerate,fft_window=512,num_of_filter=40 ,lower_freq=300,upper_freq=8000):
    
        lower_m = self.mel_scale(lower_freq)
        upper_m = self.mel_scale(upper_freq)
        dm = (upper_m - lower_m) / (num_of_filter + 1 )
        print(lower_m,upper_m,dm)

        m = []
        for i in range(num_of_filter+2):
            m.append( lower_m + i*dm)
        h = []
        for i in m:
            h.append( self.inverse_mel(i))
        f = []
        for i in h:
            #f.append( np.around((fft_window+1)*i /samplerate  ))
            f.append( np.floor((fft_window+1)*i /samplerate  ))
        
        fft_frame = []
        for frame in data:
           fft_frame.append( np.fft.rfft(frame,fft_window))

        for i in range(len(fft_frame)):
            s = abs(fft_frame[i])
            p = s*s/fft_window
            fft_frame[i] = p

        filter_bank_frames = np.zeros([len(data),num_of_filter])
        i = 0
        for frame in fft_frame:
            fbank_frame = 26*np.log(self.triangular_transform(frame,f))
            #filter_bank_frames.append(fbank_frame)
            filter_bank_frames[i] = fbank_frame
            i+=1

        #print(len(filter_bank_frames[0]),filter_bank_frames[0])
        return filter_bank_frames
        

    def abstract_mfcc(self,filter_bank_frames,num_ceps=12):
        mfcc = dct(filter_bank_frames, type=2, axis=1 , norm='ortho')[:,1:num_ceps+1]
        #print(mfcc.shape,mfcc)
        return mfcc
    
        
pp = pre_processing()
samplerate,data = sio.wavfile.read('./sa1_wav.wav')
#data,samplerate = librosa.load('./sa1.wav')

times = np.arange(len(data))/float(samplerate)
#print(0.025*samplerate)
#print(len(times),times)
#k = 0.97
#print(type(data))
emphasise_data = pp.pre_emphasise(data)
#hamming_data_made = pp.hamming_window(emphasise_data)
#hamming_data_numpy = emphasise_data*np.hamming(len(data))
#print(len(data),data)
#print(len(emphasise_data),emphasise_data)

frames=pp.frame_split(emphasise_data,samplerate)
for i in range(len(frames)):
    frames[i] = pp.hamming_window(frames[i])
#print(len(frames),frames[0])

filter_bank_frames = pp.filter_bank(frames,samplerate)
mfccs = pp.abstract_mfcc(filter_bank_frames)
#print(np.shape(filter_))
#plt.specgram(filter_bank_frames)
#x =  np.swapaxes(filter_bank_frames,0,1)
plt.matshow(mfccs)
#plt.matshow(x)
plt.xlabel('Time')
plt.ylabel('fbank')
plt.show()



'''
f1 = open("np_rfft_data.txt",'w')
f2 = open("dft_data.txt",'w')
#f3 = open("np_rfft_data_sr.txt",'w')

for i in range(len(ff)):
    f1.write(str(ff2[i])+"\n")
    f2.write(str(ff[i])+"\n")

#for i in range(len(ff3)):
#    f3.write(str(ff3[i])+"\n")


f1.close()
f2.close()
#f3.close()

scipy.io.wavfile.write("sa1_emphasise.wav",samplerate,emphasise_data)


plt.fill_between(times, data)
plt.xlim(times[0],times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.show()
'''
