import numpy as np
import matplotlib.pyplot as plt


train_data =  np.loadtxt('./data/testing_training/poly_train.csv',delimiter=',', skiprows=1)


test_data = np.loadtxt('./data/testing_training/poly_test.csv',delimiter=',', skiprows=1)
def closed_form(train_data,d,lamda):
    X_s , T_s = np.hsplit(train_data,2)
    X_s = X_s.ravel()
    T_s = T_s.ravel()
    desMat = np.vander(X_s, N=d+1, increasing=True)
    I = np.identity(d+1)
    I[0, 0] = 0
    A = desMat.T @ desMat + lamda * I
    b = desMat.T @ T_s
    wt = np.linalg.solve(A, b)
    er = 0.5*((desMat@wt-T_s).T@(desMat@wt-T_s)) + 0.5*lamda *(wt.T@wt)
    np.savetxt(
        './data/results/p2_weights_closed.csv',
        wt,
        fmt='%.18e',
        delimiter=','
        )
    
    print(er)

    return wt,er


class gradient_descent:
    def batch(train_data,d,lamda,learning_rate):
        X_s , T_s = np.hsplit(train_data,2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()
        desMat = np.vander(X_s, N=d+1, increasing=True)
        wt_closed,_ = closed_form(train_data,9,lamda)
        results = np.empty((0, 2))
       
        wt = np.zeros(d+1)
        results = np.vstack((results, [0, 0.5*((desMat@wt-T_s).T@(desMat@wt-T_s)) + 0.5*lamda *(wt.T@wt)]))
        I = np.identity(d+1)
        I[0, 0] = 0
        A = desMat.T @ desMat + lamda * I
        rel = (
            np.linalg.norm(wt - wt_closed) / np.linalg.norm(wt_closed)
        )
        itr=0
        while rel >= 1e-6 :
            # if(itr>20000): break
            delta = learning_rate*(desMat.T@(desMat@wt-T_s)+lamda*(I@wt))
            wt = wt - delta
            er = 0.5*((desMat@wt-T_s).T@(desMat@wt-T_s)) + 0.5*lamda *(wt.T@wt)
            results = np.vstack((results, [itr+1, 0.5*((desMat@wt-T_s).T@(desMat@wt-T_s)) + 0.5*lamda *(wt.T@wt)]))
            rel =np.linalg.norm(wt - wt_closed) / np.linalg.norm(wt_closed)
            print(itr+1)
            itr+=1
        print( 0.5*((desMat@wt-T_s).T@(desMat@wt-T_s)) + 0.5*lamda *(wt.T@wt))
        np.savetxt(
                './data/results/p2_weights_batch.csv',
                wt,
                fmt='%.18e',
                delimiter=','
                )
        cond_1 = np.linalg.cond(A)
        cond_2 = np.linalg.cond(desMat.T @ desMat)
        print(f"No. of Iterations: {itr}")
        print(f"cond(ΦᵀΦ): {cond_2}")
        print(f"cond(ΦᵀΦ + λI): {cond_1}")
        #The ridge penalty significantly improves the conditioning of the Hessian, 
        #reducing its condition number from 2.31e14 to 2.04e4.
        #This reduces numerical sensitivity and generally improves the convergence behavior of gradient descent.
        return wt,results

class plotting:
    def sweep(train_data,test_data):
        lams = np.logspace(-10, 2, 49)
        X_s_tr , T_s_tr = np.hsplit(train_data,2)
        X_s_tr = X_s_tr.ravel()
        T_s_tr = T_s_tr.ravel()
        X_s_ts , T_s_ts = np.hsplit(test_data,2)
        X_s_ts = X_s_ts.ravel()
        T_s_ts = T_s_ts.ravel()
        desMat_train =  np.vander(X_s_tr, 10,increasing=True)
        desMat_test  = np.vander(X_s_ts, 10,increasing=True)
        results = np.empty((0, 3))
        
        for lam in lams:
            wt , _ =closed_form(train_data,9,lam)
            rmse_ts = np.sqrt(((desMat_test @ wt - T_s_ts).T @ (desMat_test @ wt - T_s_ts)) / len(T_s_ts))
            rmse_tr = np.sqrt(((desMat_train @ wt - T_s_tr).T @ (desMat_train @ wt - T_s_tr)) / len(T_s_tr))
            results = np.vstack((results, [lam, rmse_tr,rmse_ts]))
        best_idx = np.argmin(results[:, 2])
        lambda_star = results[best_idx, 0]
        min_test_rmse = results[best_idx, 2]
        print("Optimal lambda:", lambda_star)
        print("Minimum test RMSE:", min_test_rmse)
        return lambda_star,min_test_rmse,desMat_test,X_s_ts,results,T_s_ts,X_s_tr,T_s_tr

    def plot_rmse(train_data,test_data):
        lambda_star,min_test_rmse,desMat_test,X_s_ts,results,T_s_ts,X_s_tr,T_s_tr = plotting.sweep(train_data,test_data)
        wt1, _  =closed_form(train_data,9,lambda_star)
        wt2,_ = closed_form(train_data,9,1e-10)
        wt3,_ = closed_form(train_data,9,1e-2)
        threshold = 1.05 * min_test_rmse
        valid_lams = results[results[:, 2] <= threshold, 0]
        print("5% RMSE interval:", valid_lams.min(), "to", valid_lams.max())
        print("||w(lambda_star)||2:", np.linalg.norm(wt1))
        print("||w(1e-10)||2:", np.linalg.norm(wt2))
        print("||w(1e-2)||2:", np.linalg.norm(wt3))
        fig,(ts_grp,tr_grp,grph1,grph2,grph3) = plt.subplots(1,5,figsize=(24, 5))
        tr_grp.semilogx(results[:, 0], results[:, 1])
        tr_grp.set(xlabel='lamda',ylabel='rmse_error',title='Training RMSE vs Lambda')
        tr_grp.grid()
        ts_grp.semilogx(results[:, 0], results[:, 2])
        ts_grp.set(xlabel='lamda',ylabel='rmse_error',title='Test RMSE vs Lambda')
        x_plot = np.linspace(0, 1, 500)
        Phi_plot = np.vander(x_plot, 10, increasing=True)
        true_y = np.sin(2 * np.pi * x_plot)
        for ax, wt in zip([grph1, grph2, grph3], [wt1, wt2, wt3]):
            ax.scatter(X_s_tr, T_s_tr, label='Training data')
            ax.plot(x_plot, true_y, 'r--', label='sin(2πx)')
            ax.plot(x_plot, Phi_plot @ wt, label='Ridge fit')
            ax.legend(fontsize=8)   
        ts_grp.grid()
        grph1.grid()
        grph2.grid()
        grph3.grid()
        grph1.set(xlabel='X-Axis',ylabel='Y-Axis',title ='Fitted Curve λ⋆')
        grph2.set(xlabel='X-Axis',ylabel='Y-Axis',title ='Fitted Curve λ = 1e-10')
        grph3.set(xlabel='X-Axis',ylabel='Y-Axis',title ='Fitted Curve λ = 1e-2')

        ts_grp.axhline(y=0.1, linestyle='--', label='Noise level (0.1)')
        ts_grp.legend()

        plt.show()

    
       
        
    



        



    
# gradient_descent.batch(train_data,9,0.001,0.04)#I have to change it to 0.04 when I get the real data
# # gradient_descent.stochastic(train_data,9,0.001)
# closed_form(train_data,9,0.001)
# plotting.plot_rmse(train_data,test_data)