import numpy as np

# data
X = np.array([
    [0.1, 0.2, 0.4, 0.6, 0.9],
    [0.5, 0.1, 0.3, 0.7, 0.8],
    [0.3, 0.7, 0.2, 0.1, 0.9],
    [0.9, 0.3, 0.6, 0.4, 0.2],
    [0.8, 0.5, 0.3, 0.7, 0.1],
    [0.2, 0.6, 0.5, 0.4, 0.9],
    [0.3, 0.4, 0.2, 0.1, 0.5],
    [0.6, 0.3, 0.4, 0.7, 0.2],
    [0.7, 0.1, 0.5, 0.3, 0.9],
    [0.4, 0.2, 0.6, 0.5, 0.8]
])

# target
Y = np.array([[1], [0], [1], [0], [1], [1], [0], [0], [1], [0]])

# weights 1
W1 = np.array([
    [ 0.1, -0.2],
    [ 0.4,  0.3],
    [-0.5,  0.2],
    [ 0.2, -0.3],
    [ 0.1,  0.6]
])
b1 = np.array([[0.0, 0.0]])  

# weights 2 
W2 = np.array([
    [ 0.2],
    [-0.4]
])
b2 = np.array([[0.0]])  

# params
alpha = 0.1
epochs = 100

# activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# sigmoid derivative
def DervSigmoid(a):
    return a * (1 - a)

# forward pass
for i in range(epochs):

    z1 = np.dot(X , W1) + b1 
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1 , W2) + b2
    a2 = sigmoid(z2)

    # backpropagation
    err = a2 - Y
    L = 0.5 * err ** 2

    dz2 = err * DervSigmoid(a2)
    dw2 = np.dot(a1.T , dz2)
    db2 = np.sum(dz2 , axis=0, keepdims=True)

    dz1 = np.dot(dz2 , W2.T) * DervSigmoid(a1)
    dw1 = np.dot(X.T , dz1)
    db1 = np.sum(dz1 , axis=0, keepdims=True)

    W2 = W2 - alpha * dw2
    b2 = b2 - alpha * db2
    W1 = W1 - alpha * dw1
    b1 = b1 - alpha * db1

    if i % 10 == 0:
        print(f"epoch {i} | loss : {np.mean(L)}")