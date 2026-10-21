
import numpy as np
import matplotlib.pyplot as plt

train_data = np.loadtxt('./data/testing_training/poly_train.csv',delimiter=',',skiprows=1)


def closed_form(train_data,d):
    X_s,T_s = np.hsplit(train_data,2)
    X_s = X_s.ravel()
    T_s = T_s.ravel()
    desMat = np.vander(X_s,N=d+1,increasing=True)
    wt = np.linalg.lstsq(desMat,T_s,rcond=None)[0]
    np.savetxt('./data/results/p1_weights_closed.csv',wt,fmt='%.18e',delimiter=',')
    return wt


class gradient_descent:

    def batch(train_data,d,learning_rate,epochs=2000):
        X_s,T_s = np.hsplit(train_data,2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()
        desMat = np.vander(X_s,N=d+1,increasing=True)
        Q,R = np.linalg.qr(desMat,mode="reduced")
        alpha = np.zeros(d+1)
        results = np.empty((0,2))

        for i in range(epochs):
            delta = learning_rate*(Q.T@(Q@alpha-T_s))
            alpha = alpha-delta
            er = 0.5*np.sum((Q@alpha-T_s)**2)
            results = np.vstack((results,[i+1,er]))

        wt = np.linalg.solve(R,alpha)
        np.savetxt('./data/results/p1_weights_batch.csv',wt,fmt='%.18e',delimiter=',')
        return wt,results

    def stochastic(train_data,d,learning_rate,epochs=2000):
        X_s,T_s = np.hsplit(train_data,2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()
        desMat = np.vander(X_s,N=d+1,increasing=True)
        Q,R = np.linalg.qr(desMat,mode="reduced")
        alpha = np.zeros(d+1)
        rng = np.random.default_rng(42)
        results = np.empty((0,2))

        for epoch in range(epochs):
            indices = rng.permutation(len(T_s))
            for idx in indices:
                delta = learning_rate*(Q[idx]@alpha-T_s[idx])*Q[idx]
                alpha = alpha-delta

            er = 0.5*np.sum((Q@alpha-T_s)**2)
            results = np.vstack((results,[epoch+1,er]))

        wt = np.linalg.solve(R,alpha)
        np.savetxt('./data/results/p1_weights_stochastic.csv',wt,fmt='%.18e',delimiter=',')
        return wt,results

    def mini_batch(train_data,d,learning_rate,batch_size=3,epochs=2000):
        X_s,T_s = np.hsplit(train_data,2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()
        desMat = np.vander(X_s,N=d+1,increasing=True)
        Q,R = np.linalg.qr(desMat,mode="reduced")
        alpha = np.zeros(d+1)
        rng = np.random.default_rng(42)
        results = np.empty((0,2))

        for epoch in range(epochs):
            indices = rng.permutation(len(T_s))
            for start in range(0,len(T_s),batch_size):
                idx = indices[start:start+batch_size]
                delta = learning_rate*(Q[idx].T@(Q[idx]@alpha-T_s[idx]))/len(idx)
                alpha = alpha-delta

            er = 0.5*np.sum((Q@alpha-T_s)**2)
            results = np.vstack((results,[epoch+1,er]))

        wt = np.linalg.solve(R,alpha)
        np.savetxt('./data/results/p1_weights_minibatch.csv',wt,fmt='%.18e',delimiter=',')
        return wt,results


class plotting:

    def closed_form_curves(train_data):
        X_s,T_s = np.hsplit(train_data,2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()
        x_plot = np.linspace(0,1,500)
        fig,axes = plt.subplots(3,3,figsize=(15,12))
        axes = axes.ravel()

        for d in range(1,10):
            wt = closed_form(train_data,d)
            Phi_plot = np.vander(x_plot,N=d+1,increasing=True)
            axes[d-1].scatter(X_s,T_s,label='Training data')
            axes[d-1].plot(x_plot,Phi_plot@wt,label=f'Degree {d}')
            axes[d-1].plot(x_plot,np.sin(2*np.pi*x_plot),'r--',label='sin(2πx)')
            axes[d-1].set(title=f'Degree {d}',xlabel='x',ylabel='y')
            axes[d-1].grid()
            axes[d-1].legend(fontsize=7)

        plt.tight_layout()
        plt.show()

    def loss_curves(train_data,d,learning_rate,epochs=2000):
        wt_batch,results_batch = gradient_descent.batch(train_data,d,learning_rate,epochs)
        wt_stoch,results_stoch = gradient_descent.stochastic(train_data,d,learning_rate,epochs)
        wt_mini,results_mini = gradient_descent.mini_batch(train_data,d,learning_rate,3,epochs)

        fig,ax = plt.subplots(1,1)
        ax.semilogy(results_batch[:,0],results_batch[:,1],label='Batch GD')
        ax.semilogy(results_stoch[:,0],results_stoch[:,1],label='Stochastic GD')
        ax.semilogy(results_mini[:,0],results_mini[:,1],label='Mini-batch GD')
        ax.set(xlabel='Epoch',ylabel='E(w)',title='Loss Curves')
        ax.grid(True,which='both')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def learning_rate_study(train_data,d=9):
        X_s,T_s = np.hsplit(train_data,2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()
        desMat = np.vander(X_s,N=d+1,increasing=True)
        Q,R = np.linalg.qr(desMat,mode="reduced")
        rates = [0.01,0.5,2.1]
        fig,ax = plt.subplots(1,1)

        for learning_rate in rates:
            alpha = np.zeros(d+1)
            results = np.empty((0,2))

            for i in range(100):
                delta = learning_rate*(Q.T@(Q@alpha-T_s))
                alpha = alpha-delta
                er = 0.5*np.sum((Q@alpha-T_s)**2)
                results = np.vstack((results,[i+1,er]))

            wt = np.linalg.solve(R,alpha)
            print("Learning rate:",learning_rate)
            print("Weight vector after 100 epochs:",wt)
            print("Loss after 100 epochs:",results[-1,1])
            ax.semilogy(results[:,0],results[:,1],label=f'η={learning_rate}')

        ax.set(xlabel='Epoch',ylabel='E(w)',title='Learning Rate Study')
        ax.grid(True,which='both')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def verification(train_data,d=9,learning_rate=0.1,epochs=2000):
        wt_closed = closed_form(train_data,d)
        wt_batch,_ = gradient_descent.batch(train_data,d,learning_rate,epochs)
        wt_stoch,_ = gradient_descent.stochastic(train_data,d,learning_rate,epochs)
        wt_mini,_ = gradient_descent.mini_batch(train_data,d,learning_rate,3,epochs)

        for name,wt in [('Batch GD',wt_batch),('Stochastic GD',wt_stoch),('Mini-batch GD',wt_mini)]:
            abs_error = np.linalg.norm(wt-wt_closed)
            rel_error = abs_error/np.linalg.norm(wt_closed)
            print(name)
            print("Absolute error:",abs_error)
            print("Relative error:",rel_error)


# plotting.closed_form_curves(train_data)
# plotting.loss_curves(train_data,9,0.1,2000)
# plotting.learning_rate_study(train_data,9)
# plotting.verification(train_data,9,0.1,2000)
