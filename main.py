from email.mime import audio
from microphone import Microphone
from sound_detection import load_model,infer
import soundfile as sf
import time
import numpy as np
import yamnet.yamnet


def listen(device_name, listen_window = 1):
    """
    Function that listens from the latest <listen_window> amount of seconds from
    the microphone and prints the results to console. Used for unit testing.

    Args:
        microphone (_type_): _description_
        listen_window (int, optional): _description_. Defaults to 1.
    """

    audio_window_trigger = 2
    mic = Microphone(chunksize=16000)
    model,class_names = load_model()

    # WARMUP
    for i in range(0,5):
        audio_data = mic.recorder_numpy_tf(1)
        results = infer(audio_data,model,class_names)
        #print(results)

    
    audio_frame_array = []
    audio_data = mic.stream.read(mic.stream.get_read_available())
    count = 1
    while True:
        print("Listening")
        wavfile = device_name + str(count) + ".wav"
        audio_frame_array = mic.get_all_numpy_tf(audio_frame_array)
        

        if count % audio_window_trigger == 0:
            wavfile = device_name + str(count) + ".wav"

            try: 
                sf.write(wavfile, audio_frame_array, 16000)
                #print("Audio frame array: ", audio_frame_array)
            except:
                wav_frame_array = audio_frame_array
                wav_frame_array = np.transpose(wav_frame_array)
                wav_frame_array.flatten()
                sf.write(wavfile, wav_frame_array, 16000)
                #print("Audio frame array: ", audio_frame_array)

            results = infer(audio_frame_array,model,class_names)
            print(results)
            text_file = device_name + ".txt"
            save_results(wavfile, results, text_file, count)

            audio_frame_array = []
            assert len(audio_frame_array) == 0

        count += 1
        time.sleep(0.5)

def save_results(audiofile, res, result_file, pos):
    if pos == 2:
        r = open(result_file, "w")
    else:
        r = open(result_file, "a")
    toWrite = audiofile + " inference result is " + str(res)
    r.write(toWrite)
    r.write("\n")
    r.close()




if __name__ == "__main__":
    listen("usbmic")
    #classnames = yamnet.yamnet.class_names("yamnet\\yamnet_class_map.csv")

    #print(len((classnames))