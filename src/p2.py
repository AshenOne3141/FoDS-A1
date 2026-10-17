import numpy as np

# data_P1 = np.loadtxt('../data/poly_train.csv',delimiter=',')

train_data = np.array([
    [0.000,  0.050],
    [0.111,  0.692],
    [0.222,  1.034],
    [0.333,  0.816],
    [0.444,  0.287],
    [0.556, -0.277],
    [0.667, -0.766],
    [0.778, -0.948],
    [0.889, -0.587],
    [1.000,  0.024]
])

def closed_form(train_data,d,lamda):
    X_s , T_s = np.hsplit(train_data,2)
    X_s = X_s.ravel()
    T_s = T_s.ravel()
    desMat = np.vander(X_s, N=d+1, increasing=True)
    I = np.identity(d+1)
    I[0, 0] = 0
    A = desMat.T @ desMat + lamda * I
    b = desMat.T @ T_s
    polmat = np.linalg.solve(A, b)
    np.savetxt(
        './data/p2_weights_closed.csv',
        polmat,
        fmt='%.18e',
        delimiter=','
        )

    return polmat


class gradient_descent:
    def batch(train_data,d,lamda,learning_rate):
        X_s , T_s = np.hsplit(train_data,2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()
        desMat = np.vander(X_s, N=d+1, increasing=True)
       
        wt = np.zeros(d+1)
        I = np.identity(d+1)
        I[0, 0] = 0
       
        for i in range (100000):
            delta = learning_rate*(desMat.T@(desMat@wt-T_s)+lamda*(I@wt))
            wt = wt - delta
       
        np.savetxt(
                './data/p2_weights_batch.csv',
                wt,
                fmt='%.18e',
                delimiter=','
                )
        return wt

 




    

gradient_descent.batch(train_data,9,0.001,0.04907159)#I have to change it to 0.04 when I get the real data
# gradient_descent.stochastic(train_data,9,0.001)
closed_form(train_data,9,0.001)
