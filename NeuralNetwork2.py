import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class LayerDense:
    
    def __init__(self, n_inputs, n_neurons):
        
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
       
       self.output = np.dot(inputs, self.weights) + self.biases
       
class Activation_ReLU:

   def forward(self,inputs):
       self.output = (np.maximum(0, inputs))
       
class Activation_Softmax:
      
    def forward(self, inputs):
        
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
x, y  = spiral_data(100, 3) 

print(x[:5])

dense1 = LayerDense(2, 3) 
print(dense1)

activation1 = Activation_ReLU()

dense2 = LayerDense(3, 3)

print(dense2)

activation2 = Activation_Softmax()

dense1.forward(x)

print(dense1.output[:5])

activation1.forward(dense1.output)

print(activation1.output[:5])

dense2.forward(activation1.output)

print(dense2.output[:5])

activation2.forward(dense2.output)

print(activation2.output[:5])

     