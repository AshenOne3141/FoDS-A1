
import numpy as np
import matplotlib.pyplot as plt

train_data5 = np.loadtxt('./data/testing_training/p4_train_N5.csv',delimiter=',', skiprows=1)
train_data10 = np.loadtxt('./data/testing_training/p4_train_N10.csv',delimiter=',', skiprows=1)
train_data20 = np.loadtxt('./data/testing_training/p4_train_N20.csv',delimiter=',', skiprows=1)
train_data50 = np.loadtxt('./data/testing_training/p4_train_N50.csv',delimiter=',', skiprows=1)


class bayesian_reg:

    def fit(train_data, noise, precision, deg):
        X_s, T_s = np.hsplit(train_data, 2)
        X_s = X_s.ravel()
        T_s = T_s.ravel()

        desMat = np.vander(X_s, N=deg+1, increasing=True)
        I = np.identity(deg+1)

        SN_inv = precision*I + noise*(desMat.T @ desMat)
        SN = np.linalg.solve(SN_inv, I)

        mN = noise*SN @ desMat.T @ T_s

        return X_s, T_s, desMat, SN, mN

    def predict(x, SN, mN, noise, deg):
        Phi_x = np.vander(x, N=deg+1, increasing=True)

        mean = Phi_x @ mN

       
        s2 = (1/noise) + np.sum((Phi_x @ SN) * Phi_x, axis=1)

        s = np.sqrt(np.maximum(s2, 0))

        return mean, s


class plotting:

    def plot_predictive(train_data, noise, precision, deg):
        X_s, T_s, desMat, SN, mN = bayesian_reg.fit(
            train_data, noise, precision, deg
        )

        x_plot = np.linspace(0, 1, 200)
        mean, s = bayesian_reg.predict(x_plot, SN, mN, noise, deg)

        true_y = np.sin(2*np.pi*x_plot)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        ax.scatter(X_s, T_s, label='Training data')
        ax.plot(x_plot, true_y, 'r--', label='sin(2πx)')
        ax.plot(x_plot, mean, label='Bayesian predictive mean')

        ax.fill_between(
            x_plot,
            mean - 1.96*s,
            mean + 1.96*s,
            alpha=0.2,
            label='95% predictive interval'
        )

        ax.set(xlabel='X-Axis', ylabel='Y-Axis',title='Bayesian Predictive Plot (N=10)')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        plt.show()

        return SN, mN

    def equivalence(train_data, noise, precision, deg):
        X_s, T_s, desMat, SN, mN = bayesian_reg.fit(
            train_data, noise, precision, deg
        )

        lamda = precision/noise
        I = np.identity(deg+1)

        wMAP = np.linalg.solve(
            desMat.T @ desMat + lamda*I,
            desMat.T @ T_s
        )

        print("Lambda for Ridge:", lamda)
        print("||wMAP - mN||2:", np.linalg.norm(wMAP-mN))

        x_plot = np.linspace(0, 1, 200)
        Phi_plot = np.vander(x_plot, N=deg+1, increasing=True)

        true_y = np.sin(2*np.pi*x_plot)
        mean = Phi_plot @ mN
        yMAP = Phi_plot @ wMAP

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        ax.scatter(X_s, T_s, label='Training data')
        ax.plot(x_plot, true_y, 'r--', label='sin(2πx)')
        ax.plot(x_plot, mean, label='Bayesian mean')
        ax.plot(x_plot, yMAP, ':', label='MAP / Ridge fit')

        ax.set(xlabel='X-Axis', ylabel='Y-Axis',
               title='Bayesian Mean vs MAP Fit')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_width(noise, precision, deg):
        train_datasets = {
            5: train_data5,
            10: train_data10,
            20: train_data20,
            50: train_data50
        }

        x_plot = np.linspace(0, 1, 200)

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))

        for N, train_data in train_datasets.items():
            X_s, T_s, desMat, SN, mN = bayesian_reg.fit(
                train_data, noise, precision, deg
            )

            mean, s = bayesian_reg.predict(
                x_plot, SN, mN, noise, deg
            )

            width = 2*1.96*s
            ax.plot(x_plot, width, label=f'N={N}')

        noise_floor = 2*1.96/np.sqrt(noise)

        ax.axhline(
            y=noise_floor,
            linestyle='--',
            label='Irreducible noise floor'
        )

        ax.set(xlabel='X-Axis', ylabel='95% Predictive Interval Width',
               title='Predictive Interval Width vs N')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        plt.show()


noise = 100.0       
precision = 0.05
deg = 3


# plotting.plot_predictive(train_data10, noise, precision, deg)

# plotting.equivalence(train_data10, noise, precision, deg)


# plotting.plot_width(noise, precision, deg)
