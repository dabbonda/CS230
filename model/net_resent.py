
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


#from sklearn import metrics


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:
        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags
        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        resnet18 = models.resnet18(pretrained=True)
        num_ftrs = resnet18.fc.in_features
        modules = list(resnet18.children())[:-1]
        #modules = list(resnet18.children())[:]
        self.cnn = nn.Sequential(*modules)
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.fc=nn.Linear(num_ftrs, 2)
        self.init_weights()
        self.dropout_rate = params.dropout_rate

    def init_weights(self):
        """Initialize the weights.
        """
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 224 x 224 .
        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
        Note: the dimensions after each step are provided
        """
        temps=[]
        #                                                  -> batch_size x 1 x 224 x 224
        
        #import pdb; pdb.set_trace()
        for i in np.arange(s.shape[-1]):
        #forward all slices in a series
            temp = self.cnn(s[:,:,:,:,i])
            #temp = F.avg_pool2d(temp,2)
            temp=torch.squeeze(temp)
            temps.append(temp)
        
        
        s=torch.stack(temps, 0)
            
        
        s=torch.mean(s, 0, keepdim=True)
            
            
        #average cnn output

        #s = F.dropout(s, p=self.dropout_rate, training=self.training)

       
        
        #s=torch.squeeze(s)
        
        
        s = self.fc(s)

        if len(s.shape)==3:
            s=torch.squeeze(s)
        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        #return F.log_softmax(s, dim=1)
       
        return s


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable)  batch_size dimension x 2 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    
   
    
    weights=torch.cuda.FloatTensor(2)
    weights[1]=0.25
    weights[0]=0.75
    
    #import pdb; pdb.set_trace()
    
    labels=torch.squeeze(labels)    
    
    loss=  F.cross_entropy(outputs, labels, weight=weights )
    return loss
    
    
    
    


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 2 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]
    Returns: (float) accuracy in [0,1]
    """
    #print(labels)
    #print(outputs)
    #import pdb; pdb.set_trace()
    #outputs=np.squeeze(outputs)
    outputs = np.argmax(outputs, axis=1)
  
    
    return np.sum(outputs==labels)/float(labels.size)





def AUROC(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 2 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]
    Returns: (float) accuracy in [0,1]
    """
    #print(labels)
    #print(outputs)
    #import pdb; pdb.set_trace()
    #outputs=np.squeeze(outputs)
    #outputs = np.argmax(outputs, axis=1)
    #print(outputs)
    #return np.sum(outputs==labels)/float(labels.size)
    fpr, tpr, thresholds = metrics.roc_curve(labels, outputs, pos_label=2)
   
    return metrics.auc(fpr, tpr)
    
def F1(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 2 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]
    Returns: (float) accuracy in [0,1]
    """
    #print(labels)
    #print(outputs)
    #import pdb; pdb.set_trace()
    #outputs=np.squeeze(outputs)
    outputs = np.argmax(outputs, axis=1)
    #print(outputs)
    
    return np.sum(outputs==labels)/float(labels.size)

def Confusion_Matrix(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 2 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]
    Returns: (float) accuracy in [0,1]
    """
    #print(labels)
    #print(outputs)
    #import pdb; pdb.set_trace()
    #outputs=np.squeeze(outputs)
    outputs = np.argmax(outputs, axis=1)
    #print(outputs)
    
    return np.sum(outputs==labels)/float(labels.size)


def Precision_Recall_F1(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 2 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]
    Returns: (float) accuracy in [0,1]
    """
    #print(labels)
    #print(outputs)
    #import pdb; pdb.set_trace()
    #outputs=np.squeeze(outputs)
    #outputs = np.argmax(outputs, axis=1)
    #outputs = np.argmax((outputs.data).cpu().numpy(), axis=1)
    outputs = np.argmax((outputs.data), axis=1)

    
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

    
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=2*((precision*recall)/(precision+recall))
    
    return (precision, recall,F1)

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    #'AUCROC': AUROC,
    #'F1': F1,
    'Confusion_Matrix':Confusion_Matrix,
    #'Precision_Recall': Precision_Recall,
    
    # could add more metrics such as accuracy for each token type
}

  