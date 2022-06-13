
import pyaudio
import wave
import logging
import numpy as np
import soundfile as sf

class Microphone:
    """Class representing the microphone device.
    """

    class FormatNotSupportedError(Exception):
        pass


    def __init__(self,chunksize):
        """[summary]

        Args:
            chunksize ([type]): [description]
        """
        try:
            self.CHANNELS = 2
            self.SAMPLERATE = 16000
            self.CHUNK = chunksize
            self.FORMAT = pyaudio.paInt16
            self.frames = []
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(format=self.FORMAT,channels=self.CHANNELS,rate=self.SAMPLERATE,input=True,frames_per_buffer=self.CHUNK)
            logging.info("Microphone successfully started")

        except Exception as e:
            self.CHANNELS = 1
            self.SAMPLERATE = 16000
            self.CHUNK = chunksize
            self.FORMAT = pyaudio.paInt16
            self.frames = []
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(format=self.FORMAT,channels=self.CHANNELS,rate=self.SAMPLERATE,input=True,frames_per_buffer=self.CHUNK)
            logging.info("Switching to mono channel microphone")
            logging.info("Microphone successfully started")


    def recorder_numpy_tf(self,listen_window):

        # Clear Buffer
        self.stream.read(self.stream.get_read_available())

        first = True
        for i in range(0,int(self.SAMPLERATE/self.CHUNK * listen_window)):
            data = self.stream.read(self.CHUNK)
            frame = np.frombuffer(data,dtype=np.int16)
            frame = frame/32768.0
            if self.CHANNELS == 2:
                frame = np.stack((frame[1::2],frame[::2]),axis=0)
            if first:
                frames_array = frame
                first = False
            else:
                frames_array = np.concatenate((frames_array,frame),axis=1)
        #print("Audio Frame: {}  {}".format(type(frames_array),frames_array.shape))
        return frames_array


    def get_all_numpy_tf(self,prev_audio_frame_array):
        """
        Gets all currently available audio data without having to wait.
        """
        audio_data = self.stream.read(self.stream.get_read_available())
        #print(audio_data)
        audio_frame = np.frombuffer(audio_data,dtype=np.int16)
        #wavframe = []
        #wavframe = wavframe.append(audio_frame)
        audio_frame = audio_frame/32768.0 #Normalize for yamnet 
        #print(audio_frame)
        
        if self.CHANNELS == 2:
            #print("Interleave")
            # Interleave stereo data into 2D numpy array
            audio_frame = np.stack((audio_frame[1::2],audio_frame[::2]),axis=0)
            #print(audio_frame)

        if len(prev_audio_frame_array) ==0:
            #print("First")
            audio_frame_array = audio_frame

        else:
            #audio_frame_array = np.concatenate((prev_audio_frame_array,audio_frame),axis=0)
            #audio_frame_array = np.concatenate((prev_audio_frame_array,audio_frame),axis=1)
            '''
            print("Axis 1 ", str(audio_frame_array.ndim))
            print("wavframe type: ", wavframe)
            self.save_audio(wavframe, filename)
            '''
            try:
                audio_frame_array = np.concatenate((prev_audio_frame_array,audio_frame),axis=0)
                print("Axis 0 ", str(audio_frame_array.ndim))
            except:
                #audio_frame_array = np.concatenate((prev_audio_frame_array,audio_frame),axis=1)
                audio_frame_array = np.concatenate((prev_audio_frame_array,audio_frame),axis=1)
                print("Axis 1 ", str(audio_frame_array.ndim))
                
            
        
        #print("Audio Frame: {} {}".format(type(audio_frame_array),audio_frame_array.shape))
        
        return audio_frame_array

    def save_audio(self, wav_audio, filename):
        '''
        self.frames = []
        for i in range(0,int(self.SAMPLERATE/self.CHUNK * listen_window)):
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)
        '''
        with wave.open(filename + ".wav","wb") as wavefile:
            wavefile.setframerate(self.SAMPLERATE)
            wavefile.setnchannels(self.CHANNELS)
            wavefile.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            #wavefile.writeframes(b''.join(wav_audio))
            return filename