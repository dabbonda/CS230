#calculate different metrics



import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm


import model.net as net


def final_metrics(outputs, labels):
    """
    Compute the metrics, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 2 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]
    Returns: (float) precision, recall,F1, accuracy in [0,1]
    """
    #print(labels)
    #print(outputs)
    #import pdb; pdb.set_trace()
    #outputs=np.squeeze(outputs)
    #outputs = np.argmax(outputs, axis=1)
    #outputs = np.argmax((outputs.data).cpu().numpy(), axis=1)
    #x =       np.argmax((all_outputs.data).cpu().numpy(), axis=1)

   
    #outputs = np.argmax((outputs.data), axis=1)
    #import pdb; pdb.set_trace()
    
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
   
    
    for i in range(len(outputs)): 
        if labels[i]==outputs[i]==1:
            TP += 1
        if outputs[i]==1 and labels[i]!=outputs[i]:
            FP += 1
        if labels[i]==outputs[i]==0:
            TN += 1
        if outputs[i]==0 and labels[i]!=outputs[i]:
            FN += 1

    
    if (TP+FP!=0):
        precision=TP/(TP+FP)
    else:
         precision=0   
    if (TP+FN!=0):
        recall=TP/(TP+FN)
    else:
        recall=0
    if (precision+recall!=0):
        F1=2*((precision*recall)/(precision+recall))
    else:
        F1=0
    accuracy=np.sum(outputs==labels)/float(labels.size)
    return (precision, recall,F1, accuracy)