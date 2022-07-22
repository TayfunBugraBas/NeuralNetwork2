import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class LayerDense:
    
    def __init__(self, n_inputs, n_neurons):
        
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def matrixDot(self, inputs):
       
       self.output = np.dot(inputs, self.weights) + self.biases
       
class Activation_ReLU:

   def ActReLU(self,inputs):
       self.output = (np.maximum(0, inputs))
       
class Activation_Softmax:
      
    def ActSoftmax(self, inputs):
        
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
        
class Classic_Loss:
   def calculate(self,output,y):
               
       sample_losses = self.ccel(output, y)
       data_loss = np.mean(sample_losses)
       return data_loss
        
class CCELoss(Classic_Loss):
   
    def ccel(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7,1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        
        
        
x, y  = spiral_data(100, 3) 

print(x[:5])

dense1 = LayerDense(2, 3) 
print(dense1)

activation1 = Activation_ReLU()

dense2 = LayerDense(3, 3)

print(dense2)

activation2 = Activation_Softmax()

loss_function = CCELoss()

dense1.matrixDot(x)

print(dense1.output[:5])

activation1.ActReLU(dense1.output)

print(activation1.output[:5])

dense2.matrixDot(activation1.output)

print(dense2.output[:5])

activation2.ActSoftmax(dense2.output)

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)
print(loss)

     