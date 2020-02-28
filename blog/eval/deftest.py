import pyaudio
import scipy.io.wavfile as wav
import numpy as np
import time
from scipy import signal

class Deftest:


    def testkansu(self, str):
        print(str)

    def AddOne(self,num):
        return num + 1

    def time_file_name(self,fname):
        return time.strftime("Res/%d_%b_%Y_", time.gmtime()) + fname + '.wav'

    def Audio(self,dummynum):
        #record and write
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 3
        s1 = np.fromstring(b''.join(self.recording(3)), np.int16)

        wav.write(self.time_file_name("noguchi2"), RATE, s1)
        return 1

    def Rockon(self,name):
        s2 = wav.read(self.time_file_name("noguchi2"))[1]
        # チューニングパラメータの設定
        StartCellNum = 30000
        CalLength = 20000
        WindowNum = 0
        ##FFTの実行
        FFTData = self.FFT_Func(s2, StartCellNum, CalLength, WindowNum)
        # 絶対値(FFTAmp)に変換
        FFTAmp = np.abs(FFTData)
        return FFTAmp


        # 録音用の関数を定義
    def recording(seif,sec):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = sec
        audio = pyaudio.PyAudio()

        # start Recording
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        return frames

    def FFT_Func(self,data, StartCellNum, CalLength, WindowNum):
        # FFTをかけるデータの設定
        Caldata = data[StartCellNum:StartCellNum + CalLength]
        # 窓関数の選択：WindowNum →0: ハニング窓,1:ハミング窓,Others:ブラックマン窓
        if WindowNum == 0:
            w = signal.hann(CalLength)
        elif WindowNum == 1:
            w = signal.hamming(CalLength)
        else:
            w = signal.blackman(CalLength)
        ##FFT計算
        FFTReturn = np.fft.fft(w * Caldata)

        return FFTReturn


if __name__ == '__main__':
    test = Deftest()
    test.Audio(2)
