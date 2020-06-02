#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import keras
from keras.datasets import mnist


# In[5]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
shape_x = x_train.shape
shape_y = y_train.shape
shape_x_test = x_test.shape
shape_y_test = y_test.shape

print("Shape of x: {}, Shape of y: {}".format(shape_x, shape_y))
print("Shape of x: {}, Shape of y: {}".format(shape_x_test, shape_y_test))


# In[6]:


x_test_new = []
for i in range (100):
    x_test_new.append(x_test[i].reshape(784))

x_test = np.asarray(x_test_new)

#%%

x_new = []
for i in range (32000):
    x_new.append(x_train[i].reshape(784))

x_train = np.asarray(x_new)


# In[7]:


def one_hot_encoder(label):
    l = np.zeros(10)
    l[label] = 1
    return l


# In[8]:


y_train_one_hot = np.array([one_hot_encoder(label) for label in y_train])
print("y_train {}, y_train.shape {}".format(y_train, y_train.shape))
print("y_train_one_hot.shape: {}".format(y_train_one_hot.shape))

#%%

y_test_one_hot = np.array([one_hot_encoder(label) for label in y_test])
print("y_test: {}. y_test.shape: {}".format(y_test, y_test.shape))
print("y_test_one_hot.shape: {}".format(y_test_one_hot.shape))




# In[9]:


x_train = np.array(x_new)
print("x_train.shape {} ".format(x_train.shape))
x_train = x_train/255.0



#%%

x_test = np.array(x_test_new)
print("x_test.shape {} ".format(x_test.shape))
x_test = x_test/255.0


# In[10]:


input_dim = 784
hidden_dim = 64
output_dim = 10
learning_rate = 0.1
batch_size = 16
n_epochs = 1


# In[11]:


# activation function:
def sigm_activation(x):
    sigm = 1/(1 +np.exp(-x))
    return sigm
def d_sigm_activation(x):
    d_sigm = np.exp(-x)/((1+np.exp(-x))**2)
    return d_sigm


# In[12]:


class Neural_Net:
    def __init__(self, input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, learning_rate = learning_rate):
        self.input_dim = input_dim
        self.output = output_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
        self.W2 = np.random.randn(output_dim, hidden_dim ) / np.sqrt(hidden_dim)
        self.first_W1 = self.W1
    
    def forward(self, x):
        z1 = self.W1.dot(x)
        a1 = sigm_activation(z1)
        z2 = self.W2.dot(a1)
        a2 = sigm_activation(z2)
        return(z1, a1, z2, a2)

    def weight_update(self, dW1, dW2):
        self.W2 += (-learning_rate) * dW2
        self.W1 += (-learning_rate) * dW1
      

# In[13]:


def MSE(y, output):
    mse = (y-output)**2
    #print(y, output)
    return(mse)
def d_MSE(y, output):
    d_mse = -2*y + 2*output
    return (d_mse)


# In[40]:


def Backprop(net, x, y):
    # forward
    z1, a1, z2, a2 = net.forward(x)
    # error
    error = MSE(y, a2)
    # The general error:
    g_error = error.sum()
    g_error = g_error/g_error.size
    
    # First part of derivative, de/da2:
    d3 = d_MSE(y, a2)
    # next part:
    d2 = d3 * d_sigm_activation(a2)
    dW2 = np.ones([len(d2), len(a1)])
    for i in range(0, len(d2)):
        for j in range(0, len(a1)):
            dW2[i][j] = d2[i]*a1[j]
    W2 = net.W2
    W1 = net.W1
    np.savetxt("./first_W1.txt", net.first_W1, fmt="%s")
    if (net.first_W1[30][30] != W1[30][30]):
        print("sth changed")
        np.savetxt("./output{}.txt".format(i), net.first_W1, fmt="%s")
    
    # potato = de/d(output of each neuron in hl) (sth like d3_1)
    potato = []
    for j in range (len(d2)):
        potato.append(d2[j]* W2[j, :])
    potato = np.array([potato[0], potato[1]])
    potato = potato.sum(axis = 0)
    
    # derivative of activation function of hl:
    d2_1 = potato * d_sigm_activation(a1)
    
    dW1 = np.ones([len(d2_1), len(x)])
    for i in range(0, len(d2_1)):
        for j in range(0, len(x)):
            dW1[i][j] = d2_1[i]*x[j]
    
    i +=1
        
    return(dW1, dW2, g_error)


# In[41]:

def train(x, y, batch_size = batch_size, epochs = n_epochs):
    net = Neural_Net()
    i = 0
    batch_samples = 0
    g_dW1 = np.zeros((hidden_dim, input_dim))
    g_dW2 = np.zeros((output_dim, hidden_dim))
    g_error = 0
    min_error = 1
    i_of_min_er = 0
    i_epoch = 0
    best_W1 = net.W1
    best_W2 = net.W2
    no_batch = 0
    for i in range (n_epochs):
        for j in range(len(x)):
            dW1, dW2, error = Backprop(net, x[i],y[i])
            #Batch implementation:
            batch_samples +=1
            g_dW1 += dW1
            g_dW2 += dW2
            g_error += error
    
            if (j+1) % batch_size == 0:
                no_batch +=1
                g_dW1 = g_dW1/batch_size
                g_dW2 = g_dW2/batch_size
                g_error = g_error/batch_size
                if g_error < min_error:
                    min_error = g_error
                    i_of_min_er = no_batch 
                    i_epoch = i
                    best_W1 = net.W1
                    best_W2 = net.W2
                net.weight_update(g_dW1, g_dW2)
                print("Batch no:", no_batch)
                print(g_error)
                #zeroing:
                batch_samples = 0
                g_dW1 = np.zeros((hidden_dim, input_dim))
                g_dW2 = np.zeros((output_dim, hidden_dim))
                g_error = 0
            
                print("Min error from batch {} : {}".format(no_batch, min_error))
    print("Minimal error: {}, happened in epoch no. {}, batch num: {}".format(min_error, i_epoch, i_of_min_er))
    
    np.savetxt("best_W1.txt", best_W1, fmt="%s")
    np.savetxt("best_W2.txt", best_W2, fmt="%s")
    #np.savetxt("min_error.txt", min_error.toarray(), fmt="%s")
    #np.savetxt("min_error_epoch.txt", i_epoch.toarray(), fmt="%s")
    #np.savetxt("min_error_batch.txt", i_of_min_er.toarray(), fmt="%s")
    return min_error, net

# In[42]:

min_error, our_net = train(x_train, y_train_one_hot, batch_size)

# In[101]:

# test:
    
def Test(x, y, net):
    print(y.shape)
    guessed = 0
    for i in range (0, x.shape[0]):
        _, _, _, output = net.forward(x[i])
        guessed_result = np.where(output == np.amax(output))
        correct_result = np.where(y[i] == np.amax(y[i]))
        if guessed_result == correct_result:
            guessed +=1
    accuracy = guessed/x.shape[0]
    print (accuracy)
    return accuracy
        
        
    
#%%
ytest = y_test_one_hot[:100 :]
  
test = Test(x_test, ytest, our_net) 