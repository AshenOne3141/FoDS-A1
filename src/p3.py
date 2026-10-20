import numpy as np
import matplotlib.pyplot as plt
# data_P1 = np.loadtxt('../data/poly_train.csv',delimiter=',')

train_data =  np.loadtxt('./data/testing_training/poly_train.csv',delimiter=',', skiprows=1)


test_data = np.loadtxt('./data/testing_training/poly_test.csv',delimiter=',', skiprows=1)

class gradient_descent:
    def batch(train_data,d,lamda,learning_rate):
        X_s , T_s = np.hsplit(train_data,2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()
        desMat = np.polynomial.legendre.legvander(2*X_s- 1, 9)
        results = np.empty((0, 2))
        wt = np.zeros(d+1)
        I = np.identity(d+1)
        I[0, 0] = 0
        results = np.vstack((results, [0, 0.5*((desMat@wt-T_s).T@(desMat@wt-T_s)) + lamda * np.sum(np.abs(I @ wt))]))
       
        
        for i in range (20000):
            delta = learning_rate*(desMat.T@(desMat@wt-T_s)+lamda*np.sign((I@wt)))
            wt = wt - delta
            er = 0.5*((desMat@wt-T_s).T@(desMat@wt-T_s)) + lamda * np.sum(np.abs(I @ wt))
            results = np.vstack((results, [i+1, er]))

                        
        print( 0.5*((desMat@wt-T_s).T@(desMat@wt-T_s)) + lamda * np.sum(np.abs(I @ wt)))
        np.savetxt(
                './data/results/p3_weights_batch.csv',
                wt,
                fmt='%.18e',
                delimiter=','
                )
        return (wt,results)

class plotting:
    def plot_rmse(train_data,test_data):
        lams = np.logspace(-4, 0, 21)
        X_s_tr , T_s_tr = np.hsplit(train_data,2)
        X_s_tr = X_s_tr.ravel()
        T_s_tr = T_s_tr.ravel()
        X_s_ts , T_s_ts = np.hsplit(test_data,2)
        X_s_ts = X_s_ts.ravel()
        T_s_ts = T_s_ts.ravel()
        desMat_train =  np.polynomial.legendre.legvander(2*X_s_tr- 1, 9)
        desMat_test  = np.polynomial.legendre.legvander(2*X_s_ts- 1, 9)
        results = np.empty((0, 3))

        for lam in lams:
            wt , _ =gradient_descent.batch(train_data,9,lam,0.05)
            rmse_ts = np.sqrt(((desMat_test @ wt - T_s_ts).T @ (desMat_test @ wt - T_s_ts)) / len(T_s_ts))
            rmse_tr = np.sqrt(((desMat_train @ wt - T_s_tr).T @ (desMat_train @ wt - T_s_tr)) / len(T_s_tr))
            results = np.vstack((results, [lam, rmse_tr,rmse_ts]))
        fig,(ts_grp,tr_grp) = plt.subplots(1,2)
        tr_grp.semilogx(results[:, 0], results[:, 1])
        tr_grp.set(xlabel='lamda',ylabel='rmse_error')
        tr_grp.grid()
        ts_grp.semilogx(results[:, 0], results[:, 2])
        ts_grp.set(xlabel='lamda',ylabel='rmse_error')
        ts_grp.grid()
        plt.axhline(y=0.1, linestyle='--', label='Noise level (0.1)')
      
        best_idx = np.argmin(results[:, 2])
        lambda_star = results[best_idx, 0]
        min_test_rmse = results[best_idx, 2]
        print("Optimal lambda:", lambda_star)
        print("Minimum test RMSE:", min_test_rmse)
    
    
    def plot_conv(train_data,test_data):
        gradient_descent.batch(train_data,9,0.1,0.05)
        
            
            




        


plotting.plot_rmse(train_data,test_data)

# gradient_descent.batch(train_data,9,0.1,0.05)
# gradient_descent.stochastic(train_data,9,0.001)

