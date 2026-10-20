import numpy as np
#I still have to implement Mini GD
# data_P1 = np.loadtxt('../data/poly_train.csv',delimiter=',')

train_data = np.loadtxt('./data/poly_train.csv',delimiter=',', skiprows=1)


def closed_form(train_data,d):
    X_s , T_s = np.hsplit(train_data,2)
    X_s = X_s.ravel()
    T_s = T_s.ravel()
    desMat = np.vander(X_s, N=d+1, increasing=True)
    polmat = np.linalg.lstsq(desMat, T_s, rcond=None)[0]
    np.savetxt(
        './data/p1_weights_closed.csv',
        polmat,
        fmt='%.18e',
        delimiter=','
        )

    return polmat


class gradient_descent:
    def batch(train_data,d,learning_rate):
        X_s , T_s = np.hsplit(train_data,2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()
        desMat = np.vander(X_s, N=d+1, increasing=True)
        Q, R = np.linalg.qr(desMat,mode="reduced")
        alpha = np.zeros(d+1)
        for i in range (5000):
            delta = learning_rate*(Q.T @ (Q@alpha - T_s))
            alpha = alpha - delta
        wt = np.linalg.solve(R, alpha)
        np.savetxt(
                './data/p1_weights_batch.csv',
                wt,
                fmt='%.18e',
                delimiter=','
                )
        return wt

    def stochastic(train_data,d,learning_rate):
        X_s , T_s = np.hsplit(train_data,2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()
        desMat = np.vander(X_s, N=d+1, increasing=True)
        Q, R = np.linalg.qr(desMat,mode="reduced")
        alpha = np.zeros(d+1)
        rng = np.random.default_rng(42)
        for epoch in range(5000):
            indices = rng.permutation(len(T_s))
            for idx in indices:
                delta = learning_rate * (Q[idx] @ alpha - T_s[idx]) * Q[idx]
                alpha = alpha - delta
        wt = np.linalg.solve(R, alpha)
        np.savetxt(
                './data/p1_weights_stochastic.csv',
                wt,
                fmt='%.18e',
                delimiter=','
                )
        return wt





    

gradient_descent.batch(train_data,9,0.001)
gradient_descent.stochastic(train_data,9,0.001)
closed_form(train_data,9)
