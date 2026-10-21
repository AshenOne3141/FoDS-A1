
import numpy as np
import matplotlib.pyplot as plt
import p2 as p2

train_data = np.loadtxt('./data/testing_training/poly_train.csv',delimiter=',',skiprows=1)
test_data = np.loadtxt('./data/testing_training/poly_test.csv',delimiter=',',skiprows=1)


def closed_form_ridge_leg(train_data,d,lamda):
    X_s,T_s = np.hsplit(train_data,2)
    X_s = X_s.ravel()
    T_s = T_s.ravel()
    desMat = np.polynomial.legendre.legvander(2*X_s-1,d)
    I = np.identity(d+1)
    I[0,0] = 0
    A = desMat.T@desMat+lamda*I
    b = desMat.T@T_s
    wt = np.linalg.solve(A,b)
    er = 0.5*np.sum((desMat@wt-T_s)**2)+0.5*lamda*(wt@wt)
    np.savetxt('./data/results/p3_weights_closed_ridge_leg.csv',wt,fmt='%.18e',delimiter=',')
    return wt,er


class gradient_descent:

    def batch(train_data,d,lamda,learning_rate):
        X_s,T_s = np.hsplit(train_data,2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()
        desMat = np.polynomial.legendre.legvander(2*X_s-1,d)
        results = np.empty((0,3))
        wt = np.zeros(d+1)
        I = np.identity(d+1)
        I[0,0] = 0
        best_er = np.inf

        for i in range(20000):
            delta = learning_rate*(desMat.T@(desMat@wt-T_s)+lamda*np.sign(I@wt))
            wt = wt-delta
            er = 0.5*np.sum((desMat@wt-T_s)**2)+lamda*np.sum(np.abs(I@wt))
            best_er = min(best_er,er)
            results = np.vstack((results,[i+1,er,best_er]))
        # print( 0.5*((desMat@wt-T_s).T@(desMat@wt-T_s)) + lamda * np.sum(np.abs(I @ wt)))
        np.savetxt('./data/results/p3_weights_batch.csv',wt,fmt='%.18e',delimiter=',')
        return wt,results


class plotting:

    def verify(train_data,lamda=0.1,learning_rate=0.05):
        wt,results = gradient_descent.batch(train_data,9,lamda,learning_rate)
        X_s = train_data[:,0]
        T_s = train_data[:,1]
        Phi = np.polynomial.legendre.legvander(2*X_s-1,9)
        r = T_s-Phi@wt

        approx_zero = np.abs(wt)<2*learning_rate*lamda
        nonzero = ~approx_zero
        correlations = Phi.T@r

        err_nonzero = np.max(np.abs(correlations[nonzero]-lamda*np.sign(wt[nonzero]))) if np.any(nonzero) else 0
        err_zero = np.max(np.abs(correlations[approx_zero])/lamda) if np.any(approx_zero) else 0

        w_star = np.array([0,-0.928,0.045,1.117,0.032,-0.201,0,-0.026,0,0])
        objective = 0.5*np.sum((Phi@wt-T_s)**2)+lamda*np.sum(np.abs(wt))

        print("Final objective:",objective)
        print("Reference objective:",0.2625)
        print("Max nonzero KKT violation:",err_nonzero)
        print("Max zero |correlation| / lambda:",err_zero)
        print("||w - w_star||2:",np.linalg.norm(wt-w_star))

        fig,ax = plt.subplots(1,1)
        ax.semilogy(results[:,0],results[:,1],label='E(w)')
        ax.semilogy(results[:,0],results[:,2],'--',label='Best-so-far E(w)')
        ax.set(xlabel='Iteration',ylabel='E(w)',title='Lasso Convergence, λ=0.1')
        ax.grid(True,which='both')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def sweep(train_data,test_data):
        lams = np.logspace(-4,0,21)
        X_s_tr = train_data[:,0]
        T_s_tr = train_data[:,1]
        X_s_ts = test_data[:,0]
        T_s_ts = test_data[:,1]
        desMat_train = np.polynomial.legendre.legvander(2*X_s_tr-1,9)
        desMat_test = np.polynomial.legendre.legvander(2*X_s_ts-1,9)
        results = np.empty((0,4))
        weights = []

        for lam in lams:
            wt,_ = gradient_descent.batch(train_data,9,lam,0.05)
            rmse_tr = np.sqrt(np.mean((desMat_train@wt-T_s_tr)**2))
            rmse_ts = np.sqrt(np.mean((desMat_test@wt-T_s_ts)**2))
            zero_count = np.sum(np.abs(wt)<2*0.05*lam)
            results = np.vstack((results,[lam,rmse_tr,rmse_ts,zero_count]))
            weights.append(wt)

        weights = np.array(weights)
        best_idx = np.argmin(results[:,2])
        lambda_star = results[best_idx,0]
        min_test_rmse = results[best_idx,2]
        zero_idx = np.where(np.abs(weights[best_idx])<2*0.05*lambda_star)[0]

        print("Lambda star:",lambda_star)
        print("Minimum test RMSE:",min_test_rmse)
        print("Approx-zero coefficient count:",len(zero_idx))
        print("Approx-zero coefficient indices:",zero_idx)

        return lambda_star,min_test_rmse,results,weights

    def plot_rmse(train_data,test_data):
        lambda_star,min_test_rmse,results,weights = plotting.sweep(train_data,test_data)

        fig,ax = plt.subplots(1,1)
        ax.semilogx(results[:,0],results[:,1],label='Training RMSE')
        ax.semilogx(results[:,0],results[:,2],label='Test RMSE')
        ax.axhline(y=0.1,linestyle='--',label='Noise level (0.1)')
        ax.set(xlabel='Lambda',ylabel='RMSE',title='RMSE vs Lambda')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        plt.show()

    def coefficient_path(train_data,test_data):
        lambda_star,min_test_rmse,results,weights = plotting.sweep(train_data,test_data)

        fig,ax = plt.subplots(1,1)
        for j in range(weights.shape[1]):
            ax.semilogx(results[:,0],weights[:,j],label=f'w{j}')

        ax.set(xlabel='Lambda',ylabel='Coefficient value',title='Lasso Coefficient Path')
        ax.grid()
        ax.legend(ncol=2)
        plt.tight_layout()
        plt.show()

        print("Approx-zero coefficient counts for each lambda:")
        for i,lam in enumerate(results[:,0]):
            print(lam,int(results[i,3]))

    def compare_ridge(train_data,test_data):
        lams = np.logspace(-4,0,21)
        X_s_tr = train_data[:,0]
        T_s_tr = train_data[:,1]
        X_s_ts = test_data[:,0]
        T_s_ts = test_data[:,1]
        Phi_train = np.polynomial.legendre.legvander(2*X_s_tr-1,9)
        Phi_test = np.polynomial.legendre.legvander(2*X_s_ts-1,9)
        I = np.identity(10)
        I[0,0] = 0
        ridge_results = np.empty((0,3))
        ridge_weights = []

        for lam in lams:
            wt,_ = closed_form_ridge_leg(train_data,9,lam)
            rmse_ts = np.sqrt(np.mean((Phi_test@wt-T_s_ts)**2))
            ridge_results = np.vstack((ridge_results,[lam,rmse_ts,np.linalg.norm(wt)]))
            ridge_weights.append(wt)

        ridge_weights = np.array(ridge_weights)
        best_idx = np.argmin(ridge_results[:,1])
        lambda_ridge = ridge_results[best_idx,0]
        wt_ridge = ridge_weights[best_idx]

        lambda_lasso,rmse_lasso,lasso_results,lasso_weights = plotting.sweep(train_data,test_data)
        lasso_idx = np.argmin(lasso_results[:,2])
        wt_lasso = lasso_weights[lasso_idx]

        print("\nMethod          Best lambda    Test RMSE    ||w||2    Count |wj|<0.01")
        print("Lasso",lambda_lasso,rmse_lasso,np.linalg.norm(wt_lasso),np.sum(np.abs(wt_lasso)<0.01))
        print("Ridge",lambda_ridge,ridge_results[best_idx,1],np.linalg.norm(wt_ridge),np.sum(np.abs(wt_ridge)<0.01))

        return lambda_lasso,wt_lasso,lambda_ridge,wt_ridge

    def plot_curves(train_data,test_data):
        lambda_lasso,wt_lasso,lambda_ridge,wt_ridge = plotting.compare_ridge(train_data,test_data)

        X_s = train_data[:,0]
        T_s = train_data[:,1]
        x_plot = np.linspace(0,1,500)
        Phi_plot = np.polynomial.legendre.legvander(2*x_plot-1,9)
        true_y = np.sin(2*np.pi*x_plot)

        fig,ax = plt.subplots(1,1)
        ax.scatter(X_s,T_s,label='Training data')
        ax.plot(x_plot,true_y,'r--',label='sin(2πx)')
        ax.plot(x_plot,Phi_plot@wt_ridge,label='Ridge fit')
        ax.plot(x_plot,Phi_plot@wt_lasso,label='Lasso fit')
        ax.set(title='Lasso vs Ridge Fitted Curves',xlabel='x',ylabel='y')
        ax.legend(fontsize=8)
        ax.grid()
        plt.tight_layout()
        plt.show()


# plotting.verify(train_data,0.1,0.05)
# plotting.plot_rmse(train_data,test_data)
# plotting.coefficient_path(train_data,test_data)
# plotting.compare_ridge(train_data,test_data)
# plotting.plot_curves(train_data,test_data)
