import yamnet.params as yamnet_params
import yamnet.yamnet as yamnet_model
import yamnet.metadata as metadata
import numpy as np
from operator import itemgetter
import time

def load_model():
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    yamnet_classes = np.array([x['name'] for x in metadata.CAT_META])
    return yamnet,yamnet_classes

def infer(audio_data,model,yamnet_classes):
    #print(type(audio_data))
    #print(audio_data[0][20])
    #print(audio_data[1][20])
    audio_data = np.transpose(audio_data) #TODO: Verify if transposition needed
    #print("After transpose: {} {} {}".format(type(audio_data),audio_data.shape,audio_data.dtype))
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data,axis=1)
    #print("After mean: {} {} {}".format(type(audio_data),audio_data.shape,audio_data.dtype))
    scores,embeddings,spectrogram = model(audio_data)

    top_N = 5
    mean_scores = np.mean(scores,axis=0)
    top_class_indices = np.argsort(mean_scores)[::-1][:top_N]

    results = []
    total_scores = 0
    for i in top_class_indices:
        pred_class = yamnet_classes[i].replace("'","")
        pred = (pred_class,mean_scores[i])
        total_scores += mean_scores[i]
        results.append(pred)

    #unknown_class_score = 1.0 - total_scores
    #results.append(("unknown",unknown_class_score))
    

    return (sorted(results,key=itemgetter(1),reverse=True))[0]


