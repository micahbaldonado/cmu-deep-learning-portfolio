import numpy as np
import matplotlib.pyplot as plt

# we want this to accept # of layers, activation functions
# (include none), # of neurons per layer
### Parameter names and layers follow the ones used in class.

def relu(x, derivative=False):
    '''
    input: x has shape (batch_size, input_dim)
    Returns either the relu activation of x or the derivative of relu of x. 
    Both have shape (batch_size, input_dim).
    '''
    if derivative:
        result = np.empty(shape=x.shape)
        result[x<=0] = 0
        result[x>0] = 1
        return result
    return np.maximum(0,x)

def sigmoid(x, derivative = False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

def tanh(x, derivative = False):
    if derivative:
        return 1 - tanh(x) * tanh(x)
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def linear(x, derivative = False):
    if derivative:
        return 1
    return x
            
#########################################
# General MLP

activation_functions = {
    "ReLU": relu,
    "Sigmoid": sigmoid,
    "Tanh": tanh,
    "Linear": linear
}

class MyMLP:
    '''
    The MLP model class that puts together forward, backward, gradient update, and preict functions.
    Parameters of the class are the weights and biases.
    '''
    
    def __init__(self, init, input_dim, hidden_dims, output_dim, activations): # hidden dims will be an array of values which correspond to the number of perceptrons per hidden layer        

        self.w = []
        self.b = []
        self.z = []
        self.h = []
        self.prev_update_w = [] # This is used down the line with the SGD with momentum
        self.prev_update_b = []


        # self.acts will be an array containing the activations per each layer. Might write them as a string.
        self.act = [activation_functions[act] for act in activations]

        def init_weights(init, input, output):
            if(init == 'He'):
                return np.random.randn(input, output) * np.sqrt(2. / input)
            elif (init == 'Xavier'):
                return np.random.randn(input, output) * np.sqrt(2. / (input + output))
            else:
                raise ValueError("'He' or 'Xavier' only")
            

        if(hidden_dims == [0]):
            self.w.append(init_weights(init, input_dim, output_dim))
            self.b.append(np.zeros(output_dim))
        else:
            # create the first layer
            self.w.append(init_weights(init, input_dim, hidden_dims[0]))
            self.b.append(np.zeros(hidden_dims[0]))

            num_hidden_layers = len(hidden_dims)
        
            for i in range(num_hidden_layers - 1): # if 2, then it will return hidden_dim[0] to hidden_dim[1]
                self.w.append(init_weights(init, hidden_dims[i], hidden_dims[i+1]))
                self.b.append(np.zeros(hidden_dims[i+1]))

            self.w.append(init_weights(init, hidden_dims[-1], output_dim))
            self.b.append(np.zeros(output_dim))

        # This is useful for momentum SGD later:
        self.velocity_w = [np.zeros_like(w) for w in self.w]
        self.velocity_b = [np.zeros_like(b) for b in self.b]

        # This is useful for ADAM later:
        self.momentum_w = [np.zeros_like(w) for w in self.w]
        self.momentum_b = [np.zeros_like(b) for b in self.b]
            
        self.variance_w = [np.zeros_like(w) for w in self.w]
        self.variance_b = [np.zeros_like(b) for b in self.b]

        self.t = 1

    def forward(self, x): 
        self.z = [] 
        self.h = [x.copy()] # The input itself IS an activation.

        for i in range(len(self.w)): 
            x = np.dot(x, self.w[i]) + self.b[i]
            self.z.append(x)
            x = self.act[i](x)
            self.h.append(x)

            # if(i % 1000 == 0):
                # print("last z layer", self.z[-1][0])
            # logic may be wrong here.

        return x
    
    def backward(self, dLdz):

        self.dLdw = [] 
        self.dLdb = [] 

        dLdz_cur = dLdz

        # We want to start at the last layer and iterate to the first layer
        for i in range(len(self.w) - 1, -1, -1): 
            dLdb = dLdz_cur
            self.dLdb.insert(0, dLdb) # We use the insert method to preserve the grad order.

            dLdw = np.dot(self.h[i].T, dLdz_cur) # Must transpose self.h for the shapes to match
            self.dLdw.insert(0, dLdw)

            if i > 0:
                dLdz_cur = np.dot(dLdz_cur, self.w[i].T)
                dLdz_cur = dLdz_cur * self.act[i-1](self.z[i-1], derivative = True)

        return self.dLdw, self.dLdb
    
    def loss_fc(self, y, y_pred, loss_type): # assumption: y is the ground truth.
        # L2 LOSS
        if(loss_type == 'L2'):
            L2_loss = np.sum((y - y_pred) ** 2.)
            # return L2_loss, -2.*(y-y_pred)*self.act[-1](self.z[-1], derivative = True)  # 2nd term dL/dz.
            # return L2_loss, -2.*(y-y_pred)*self.act[-1](self.z[-1], derivative = True)  # 2nd term dL/dz.
            return L2_loss, -2.*(y-y_pred) # This is more dL/dh. dL/dz may be calculated in the backward pass
        # BINARY CROSS ENTROPY LOSS
        elif(loss_type == 'BCE'): 
                # To prevent log(0) error, we add a clip.
                epsilon = 1e-12
                y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon) # We must ensure y_pred is between 0 and 1
                # just not exactly 0 or 1.

                loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

                dLdz = y_pred - y # This is assuming that the last layer is the sigmoid activation function
            
                return loss, dLdz
        else:
            raise ValueError('Should be L2 or CE')

    
    def predict(self, x):
        print("predict method!!!")
        z = self.forward(x) # (batch_size, output_dim)

        # y1_prob = sigmoid(z) # I commented this out because the forward pass should have a sigmoid activation function
        predictions = (z > 0.5) + 0 # (batch_size, output_dim)

        # greater than 50% is standard here.

        # print(predictions)

        return predictions
        
    def SGD_update(self, lr):        
        # perform SGD to update the parameters
        # WE want to iterate across EVERY batch and take the average
        # We take advantage of how the batch size is the first dimension.
        # The following approach doesn't work:
            # self.w -= lr * np.mean(self.dLdw, axis=0) # dLdW has shape (batch_size, input_dim, hidden_dim)
            # self.b -= lr * np.mean(self.dLdb, axis = 0) # dLdc has shape (batch_size, hidden_dim) 

        # Because I opted to store the gradients in lists, to access the averages
        # per list element in the list, I need to loop through them.
        for i in range(len(self.w)):
            dLdw_avg = np.mean(self.dLdw[i], axis=0)
            dLdb_avg = np.mean(self.dLdb[i], axis=0)

            self.w[i] -= lr * dLdw_avg
            self.b[i] -= lr * dLdb_avg

    def SGD_W_Momentum(self, lr, momentum):

        for i in range(len(self.w)):
            dLdw_avg = np.mean(self.dLdw[i], axis=0)
            dLdb_avg = np.mean(self.dLdb[i], axis=0)

            self.velocity_w[i] = momentum * self.velocity_w[i] + (1-momentum) * dLdw_avg # v_t-1 is the same as cur t in this code
            self.velocity_b[i] = momentum * self.velocity_b[i] + (1-momentum) * dLdb_avg # we use self.velocity_w[i] and b[i] to avoid rewriting the variable.
            
            self.w[i] -= lr * self.velocity_w[i]
            self.b[i] -= lr * self.velocity_b[i]

    def ADAM(self, lr, beta_1, beta_2): # beta_1 is like the momentum term
        # ADAM is useful because it adapts learning rates for each param
        # it adpats the weights and biases with different learning rates, which leads to much more ideal convergence.

        alpha = lr
        epsilon = 1e-12

        for i in range(len(self.w)):
            dLdw_avg = np.mean(self.dLdw[i], axis=0)
            dLdb_avg = np.mean(self.dLdb[i], axis=0)

            self.momentum_w[i] = beta_1 * self.momentum_w[i] + (1-beta_1) * dLdw_avg # v_t-1 is the same as cur t in this code
            self.momentum_b[i] = beta_1 * self.momentum_b[i] + (1-beta_1) * dLdb_avg # v_t-1 is the same as cur t in this code
            
            self.variance_w[i] = beta_2 * self.variance_w[i] + (1-beta_2) * dLdw_avg ** 2 # we use self.velocity_w[i] and b[i] to avoid rewriting the variable.
            self.variance_b[i] = beta_2 * self.variance_b[i] + (1-beta_2) * dLdb_avg ** 2 # we use self.velocity_w[i] and b[i] to avoid rewriting the variable.

            # bias correction: we make sure that momentum is not underestimated and that variance is not initailly too small 
            bias_corrected_momentum_w = self.momentum_w[i] / (1 - beta_1 ** self.t)
            bias_corrected_momentum_b = self.momentum_b[i] / (1 - beta_1 ** self.t)
            
            bias_corrected_variance_w = self.variance_w[i] / (1 - beta_2 ** self.t)
            bias_corrected_variance_b = self.variance_b[i] / (1 - beta_2 ** self.t)

            self.w[i] -= alpha * bias_corrected_momentum_w/ np.sqrt(bias_corrected_variance_w + epsilon)
            self.b[i] -= alpha * bias_corrected_momentum_b/ np.sqrt(bias_corrected_variance_b + epsilon)

        self.t += 1