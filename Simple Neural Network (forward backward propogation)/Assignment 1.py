import numpy as np

X = np.array([[0.2], [0.8]]) #input array with 2 inputs 
Y = np.array([[1]]) #actual output 

#sigmoid function 
def sigmoid(z):
    return 1/(1+np.exp(-z)) #using np.exp() as working with arrays hence cannot use .math lib 

#derivative of sigmoid function 
def dsig(z):
    return sigmoid(z) * (1-sigmoid(z))

def simpleNeuralNetwork(X, Y):
    epochs = 10000 #fixed number of iterations 
    learning_rate = 0.5 
    W1 = np.random.uniform(-1, 1, (3, 2)) #hidden layer weights: 3 neurons 2 inputs 
    #random number generated from -1 to 1 as sigmoid has outputs within this range 
    b1 = np.random.uniform(-1, 1, (3, 1)) #hidden layer bais: 3 neurons and each neuron has 1 bais value 
    W2 = np.random.uniform(-1, 1, (1, 3)) #output layer weights: 1 neurons 3 inputs 
    b2 = np.random.uniform(-1, 1, (1, 1)) #output layer bais: 1 neuron and each neuron has 1 bais value 

    for epoch in range(epochs):

        #1. Forward propogation 
        Z1 = np.dot(W1, X) + b1 #matrix multiplication 
        A1 = sigmoid(Z1)
        
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        
        #2. Compute loss
        Loss = -Y*np.log(A2) + (1-Y)*np.log(1-A2) #single output else sum of all outputs loss
        
        #3. Backward Propogation

        #Gradient calculations
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) #A1.T is transpose of A1 as to multiply by dZ2 (1,1) need transpose of A1 (from (3, 1) to (1, 3))
        db2 = dZ2
        
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * dsig(Z1)
        dW1 = np.dot(dZ1, X.T )
        db1 = dZ1
        
        #4. Update weights 
        #subtract from the previous value 
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
        #print after each 1000 values 
        if epoch %1000 == 0:
            print(f"Epoch {epoch}, Loss: {Loss}")

    print(f"\nFinal Output: {A2[0][0]:.6f}")


simpleNeuralNetwork(X, Y)