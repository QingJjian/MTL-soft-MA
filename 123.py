import scipy
import scipy.io.wavfile as wav
import matplotlib.pylab as pylab
import soundfile as sound
import librosa
import wave
import struct
from scipy import *
import seaborn as sns
from pylab import *
filename = 'C:/audio2019/airport-barcelona-0-0-a.wav'
stereo,sr = sound.read(filename)
stereo = np.asfortranarray(stereo)
cm = librosa.feature.melspectrogram(stereo[:,0], sr=48000, n_fft=2048, hop_length=1024, n_mels=128)
sns.set(font_scale=1.25)#字符大小设定
hm=sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})#, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

