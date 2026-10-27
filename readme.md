# Foundations of Data Science

## Assignment 1 --- Code Documentation

This document describes the supplied Python implementation for P1--P4,
following the organisation and requirements of the **CS F320:
Foundations of Data Science Assignment-1** PDF. It explains the modules,
algorithms, inputs, outputs, and how to run the menu-driven CLI.

  -----------------------------------------------------------------------
  Module                  Topic                   Main implementation
  ----------------------- ----------------------- -----------------------`p1.py`Polynomial regression   Least-squares fit;
                                                  QR-reparameterised
                                                  batch, stochastic, and
                                                  mini-batch gradient
                                                  descent; plots and
                                                  verification.`p2.py`Ridge regression        Closed-form ridge,
                                                  gradient descent,
                                                  regularisation sweep,
                                                  train/test RMSE and
                                                  fitted curves.`p3.py`Lasso regression        Legendre basis,
                                                  subgradient descent,
                                                  KKT checks, coefficient
                                                  paths, RMSE sweep,
                                                  ridge comparison.`p4.py`Bayesian regression     Posterior
                                                  mean/covariance,
                                                  predictive uncertainty,
                                                  MAP/ridge equivalence,
                                                  effect of sample size.`cli.py`Program launcher        Interactive
                                                  command-line menu to
                                                  select and run the
                                                  available tasks in
                                                  P1--P4.
  -----------------------------------------------------------------------

## 1. Project layout and data

The code expects to be run from a project folder containing the Python
files and the supplied CSV datasets. The CLI imports the modules from a
folder named`src`and searches for the CSVs in relative locations.

Recommended layout:`text
FoDS-A1/
├── cli.py
├── src/
│   ├── p1.py
│   ├── p2.py
│   ├── p3.py
│   └── p4.py
└── data/
    ├── testing_training/
    │   ├── poly_train.csv
    │   ├── poly_test.csv
    │   ├── p4_train_N5.csv
    │   ├── p4_train_N10.csv
    │   ├── p4_train_N20.csv
    │   └── p4_train_N50.csv
    └── results/`-   Each CSV is expected to contain two columns,`x`and`y`, with a
    header row. The code loads numeric rows with NumPy and skips the
    header.
-   Use the fixed provided datasets. The assignment states that data
    should not be generated or modified and that`sin(2πx)`is for
    plotting only.
-   The scripts save selected weight vectors under`data/results`.
    Create this directory before running tasks that call`np.savetxt`.
-   The CLI's P1 loader checks`data/poly_train.csv`and`../data/testing_training/poly_train.csv`relative to the project
    root; P2--P4 use`../data/testing_training`paths. Keep the folder
    layout consistent with these relative paths, or adjust the paths in`cli.py`.

## 2. Running the program

Install the required packages:`bash
pip install numpy matplotlib`From the project root, run:`bash
python cli.py`The main menu lets you select P1, P2, P3, P4, or exit. Each problem has
a submenu for running individual plots/checks or a group of tasks. Use
option`0`to return to the previous menu. Plot windows must be closed
before the program can continue to the next plotting task.

**Important import note:** keep module-level execution limited. The CLI
imports`p1.py`--`p4.py`, so data loading or plotting that should happen
only when a file is run directly should be placed inside an`if __name__ == "__main__":`block. The currently supplied`p1.py`--`p3.py`load some CSVs at import time, which means their paths
must already be valid even before a menu option is selected.

## 3. P1 --- Polynomial regression and gradient descent

### Goal and model

P1 fits polynomial models of degrees 1 through 9 to the ten training
samples. For degree (d), the monomial design matrix contains columns (1,
x, x^2, ..., x^d). The objective is`E(w) = 0.5||Phi w - t||^2.`### Methods

-   **Closed form:** builds a Vandermonde design matrix and obtains
    weights with`np.linalg.lstsq`, a stable least-squares solver.
-   **QR reparameterisation:** factors (Phi = QR) and optimises
    in (alpha)-space using (Q), then recovers (w) by solving
    (Rw=alpha). This reduces the numerical difficulty of
    degree-9 optimisation.
-   **Batch GD:** updates the parameter vector once per epoch using the
    full gradient (Q^T(Qalpha-t)).
-   **Stochastic GD:** shuffles the samples each epoch and updates from
    one sample at a time. The random generator uses seed`42`.
-   **Mini-batch GD:** shuffles samples and updates using batches of
    size 3.

### CLI options

1.  Plot closed-form fits for degrees 1--9 alongside training points and
    the true sine curve.
2.  Plot loss versus epoch for batch, stochastic, and mini-batch GD.
3.  Compare learning rates`0.01`,`0.5`, and`2.1`over 100 epochs;
    print final weights and loss for each.
4.  Compare GD weight vectors against the closed-form solution using
    absolute and relative Euclidean errors.

### Outputs

Plots include nine fitted curves, a combined log-scale loss plot, and a
learning-rate comparison. Weight vectors are written to:

-`data/results/p1_weights_closed.csv`-`data/results/p1_weights_batch.csv`-`data/results/p1_weights_stochastic.csv`-`data/results/p1_weights_minibatch.csv`## 4. P2 --- Ridge regression

### Goal and objective

P2 adds an L2 penalty to polynomial least squares. The assignment's
objective is`E(w) = 0.5||Phi w-t||^2+lambda/2||w||^2.`For degree 9 in the monomial basis, the closed-form solution is obtained
by solving`(Phi^T Phi + lambda I) w = Phi^T t`### Methods and outputs

-   **Closed form:** solves the regularised linear system with`np.linalg.solve`.
-   **Gradient descent:** starts at zero and uses the ridge gradient
    (Phi^T(Phi w-t)+lambda w). The code
    compares its weights with the closed-form weights and reports the
    iteration count and condition numbers.
-   **Regularisation sweep:** evaluates (lambda) over`np.logspace(-10, 2, 49)`, calculates training and test RMSE, and
    chooses (lambda^*) with the lowest test RMSE.
-   **Fitted curves:** compares (lambda=10^{-10}), the
    selected (lambda^*), and
    (lambda=10^{-2}) against the training data and
    (sin(2pi x)).

The CLI provides an RMSE sweep, a closed-form solution at
(lambda=0.01), and a combined option for the available P2
plots. The RMSE plot includes`0.1`as the reference noise level. The
code also prints a 5% interval of (lambda) values whose test
RMSE is no more than 1.05 times the minimum.

## 5. P3 --- Lasso using subgradient descent

### Basis and objective

P3 uses the degree-9 shifted Legendre basis (P_j(2x-1)), implemented
with`np.polynomial.legendre.legvander(2*x - 1, 9)`. This spans the same
degree-9 polynomial function class as the monomial basis but is
generally better conditioned. The Lasso objective is`E(w)=0.5||Phi w-t||^2+lambda||w||_1.`### Methods and outputs

-   **Subgradient descent:** starts from zero and performs 20,000
    updates with step size`0.05`for each (lambda) in`np.logspace(-4, 0, 21)`.
-   **Convergence and KKT check:** at (lambda=0.1), plots the
    objective and best-so-far objective and reports KKT-related
    residual-correlation checks against the provided reference objective
    and coefficient vector.
-   **RMSE sweep:** plots training and test RMSE against
    (lambda) and reports the best test-RMSE (lambda)
    and approximate-zero coefficients.
-   **Coefficient paths:** plots each coefficient versus
    (lambda) and reports approximate-zero counts for each
    value.
-   **Ridge comparison:** fits ridge regression in the Legendre basis,
    chooses each model's best (lambda) by test RMSE, and
    compares test RMSE, coefficient norm, and coefficient counts below`0.01`.
-   **Fitted curves:** overlays the selected Lasso and Ridge fits,
    training data, and true sine curve.

The assignment defines an approximate-zero coefficient by
(||w_j||\<2etalambda); with (eta=0.05), the
code uses this threshold in its sweep and coefficient-count
calculations.

## 6. P4 --- Bayesian polynomial regression

### Model and hyperparameters

P4 uses degree 3 with observation-noise precision (beta=100)
(noise variance`0.01`) and prior precision (alpha=0.05). The
posterior weight precision is`S_N^(-1) = alpha I + beta Phi^T Phi`and the code solves for covariance (S_N) and computes posterior mean`m_N=beta S_NPhi^Tt.`### Predictive distribution

For a point (x), the predictive mean is (m(x)=\phi(x)^Tm_N).
The predictive variance is`s^2(x)=beta^{-1}+\phi(x)^TS_N\phi(x),`which includes both observation noise and parameter uncertainty. The
plot shows the mean and the 95% predictive interval
(m(x)\pm1.96s(x)).

### CLI options

1.  Predictive mean and 95% interval using`p4_train_N10.csv`.
2.  Compare the Bayesian posterior mean with the MAP/Ridge estimate
    using (lambda=alpha/beta=0.0005); print
    the Euclidean difference between the weight vectors.
3.  Plot 95% predictive interval widths for (N=5,10,20,50), with the
    irreducible noise floor.

The expected trend is that the parameter-uncertainty contribution
decreases as (N) increases, while the observation-noise contribution
remains.

## 7. Function reference

| File | Function / method | Purpose |
|---|---|---|
| `p1.py` | `closed_form(train_data, d)` | Computes least-squares weights for a degree-`d` polynomial in the monomial basis. |
| `p1.py` | `gradient_descent.batch(...)` | Runs batch gradient descent using the QR-reparameterised design matrix. Returns the weights and loss history. |
| `p1.py` | `gradient_descent.stochastic(...)` | Runs stochastic gradient descent with shuffled samples. Returns the weights and loss history. |
| `p1.py` | `gradient_descent.mini_batch(...)` | Runs mini-batch gradient descent with batches of size 3. Returns the weights and loss history. |
| `p1.py` | `plotting.closed_form_curves(...)` | Plots closed-form polynomial fits for degrees 1–9. |
| `p1.py` | `plotting.loss_curves(...)` | Plots the loss histories of batch, stochastic, and mini-batch gradient descent. |
| `p1.py` | `plotting.learning_rate_study(...)` | Compares convergence for several learning rates and reports the final weights and loss. |
| `p1.py` | `plotting.verification(...)` | Compares gradient-descent weights with the closed-form weights. |
| `p2.py` | `closed_form(train_data, d, lamda)` | Computes the closed-form ridge-regression weights and objective value. |
| `p2.py` | `gradient_descent.batch(train_data, d, lamda, learning_rate)` | Runs ridge gradient descent and returns the weights and loss history; also reports conditioning information. |
| `p2.py` | `plotting.sweep(train_data, test_data)` | Evaluates the fixed regularisation grid and calculates train/test RMSE for each value. |
| `p2.py` | `plotting.plot_rmse(train_data, test_data)` | Plots RMSE against the regularisation parameter and displays fitted curves. |
| `p3.py` | `closed_form_ridge_leg(train_data, d, lamda)` | Computes a ridge-regression fit using the Legendre basis. |
| `p3.py` | `gradient_descent.batch(train_data, d, lamda, learning_rate)` | Runs Lasso subgradient iterations and returns the final weights and objective history. |
| `p3.py` | `plotting.verify(train_data, lamda, learning_rate)` | Checks the Lasso result against KKT-related conditions and the supplied reference solution. |
| `p3.py` | `plotting.sweep(train_data, test_data)` | Evaluates Lasso across the regularisation grid and records RMSE, coefficient counts, and weights. |
| `p3.py` | `plotting.plot_rmse(...)` | Plots training and test RMSE against the regularisation parameter. |
| `p3.py` | `plotting.coefficient_path(...)` | Plots each Lasso coefficient against the regularisation parameter and reports approximate-zero counts. |
| `p3.py` | `plotting.compare_ridge(...)` | Compares the best Lasso and Ridge models using test RMSE, weight norm, and small-coefficient counts. |
| `p3.py` | `plotting.plot_curves(...)` | Plots the selected Lasso and Ridge fits with the training data and true sine curve. |
| `p4.py` | `bayesian_reg.fit(train_data, noise, precision, deg)` | Computes the posterior covariance and mean for Bayesian polynomial regression. |
| `p4.py` | `bayesian_reg.predict(x, SN, mN, noise, deg)` | Computes the predictive mean and standard deviation at the requested inputs. |
| `p4.py` | `plotting.plot_predictive(...)` | Plots the predictive mean and 95% predictive interval. |
| `p4.py` | `plotting.equivalence(...)` | Compares the Bayesian posterior mean with the MAP/Ridge estimate. |
| `p4.py` | `plotting.plot_width(...)` | Compares predictive interval widths for training sets of different sizes. |

## 8. Implementation notes to check before submission

This documentation describes the supplied code, not a claim that every
detail has already been verified against every rubric item. The
following consistency checks are worth addressing before treating the
results as final:

-   **P1 mini-batch averaging:** the current mini-batch update uses the
    sum of sample gradients in each batch. The assignment specifies the
    averaged gradient over the mini-batch; divide the batch gradient by
    the batch size to match that requirement.
-   **P2 regularisation convention:**`p2.py`sets`I[0,0] = 0`, so the
    intercept is not penalised in the solved system. The assignment's
    stated formula uses (lambda I) for all coefficients. Also,
    the printed objective includes a penalty on all entries, so it is
    not fully consistent with the solved system.
-   **P3 regularisation convention:**`p3.py`sets`I[0,0] = 0`in the
    subgradient update and ridge system, while some objective and KKT
    calculations apply the penalty to all coefficients. Make the penalty
    convention consistent with the assignment formula
    (lambda||w||_1) and its KKT checks, unless your course
    instructions specify otherwise.
-   **P2 degree argument:**`gradient_descent.batch`calls`closed_form(train_data, 9, lamda)`regardless of the`d`argument,
    and the relative-error calculation has no explicit zero-denominator
    guard. This is harmless only when`d=9`and the reference norm is
    nonzero.
-   **Paths and output directory:** the CLI expects a particular project
    layout and`data/results`to exist. Verify the relative paths before
    running the menu.

These notes are included to make the documentation transparent and help
align the implementation with the assignment PDF. They do not alter the
submitted source files.



## 9. Answers to theory and discussion questions

The answers below address the conceptual checkpoints and discussion prompts in the assignment. Numerical values that depend on running the code are intentionally not fabricated; insert the values printed by your implementation where the assignment asks for measured results.

### P1 — Polynomial regression and gradient descent

**Why does the SGD loss curve coincide with the Batch GD curve?**

With the QR reparameterisation, the columns of \(Q\) are orthonormal, so \(Q^TQ=I\). Each row of \(Q\) has squared norm \(1/N\) when the square \(N\times N\) orthogonal case applies here. The individual-sample gradient updates therefore have a particularly regular structure, and the assignment expects the SGD loss curve to coincide with the batch curve under this construction. Refer to the plotted curves to confirm the behaviour in the implementation.

**Why does Mini-batch GD converge more slowly?**

Mini-batch GD uses an averaged gradient for each group of samples, so each update is based on only part of the dataset rather than the full gradient. It makes four updates per epoch for batches of sizes 3, 3, 3, and 1, while batch GD makes one full-gradient update. The resulting sequence of updates is different from batch GD, so its loss can approach the minimum at a different rate. The last batch contains one sample, so its average is simply that sample's gradient.

**What does the learning rate do?**

A small learning rate such as \(\eta=0.01\) makes stable progress but converges slowly. A suitable rate in the range specified by the assignment gives faster convergence. A rate above the stability limit, such as \(\eta=2.1\), causes the loss to diverge. At \(\eta=2\), the update is at the boundary for the orthonormal QR objective, so the loss may oscillate rather than decrease smoothly.

**Why use QR reparameterisation?**

The degree-9 monomial design matrix is severely ill-conditioned. Direct gradient descent in the original weight coordinates can converge extremely slowly because the directions have very different curvature. Writing \(\Phi=QR\) and optimising \(E(\alpha)=\frac12\|Q\alpha-t\|^2\) gives an orthonormal design matrix and a better-conditioned optimisation problem. The original weights are recovered by solving \(Rw=\alpha\).

### P2 — Ridge regression

**How does a Gaussian prior lead to Ridge regression?**

Bayes' rule gives \(p(w\mid t,X)\propto p(t\mid w,X)p(w)\). With a Gaussian likelihood of precision \(\beta\), the likelihood contributes the term \(\frac{\beta}{2}\|\Phi w-t\|^2\) to the negative log posterior. A zero-mean Gaussian prior with precision \(\alpha\) contributes \(\frac{\alpha}{2}\|w\|^2\). Dividing by \(\beta\) gives the equivalent minimisation objective
\[
\frac12\|\Phi w-t\|^2+\frac{\lambda}{2}\|w\|^2,\qquad \lambda=\frac{\alpha}{\beta}.
\]
Thus, the MAP estimate is the Ridge solution.

**How is the closed-form Ridge solution obtained?**

Differentiate the objective with respect to \(w\):
\[
\nabla \widetilde E(w)=\Phi^T(\Phi w-t)+\lambda w.
\]
Setting the gradient to zero gives
\[
(\Phi^T\Phi+\lambda I)w=\Phi^Tt.
\]
Solve this linear system directly rather than explicitly calculating a matrix inverse.

**What does Ridge regularisation do to conditioning?**

The matrix \(\Phi^T\Phi\) can be poorly conditioned for high-degree monomial features. Adding \(\lambda I\), with \(\lambda>0\), increases the eigenvalues by \(\lambda\) and generally reduces the condition number. This makes the system less sensitive to numerical errors and can make gradient descent more practical. Report the two condition numbers measured by your code.

**How should \(\lambda\) affect the fit and errors?**

When \(\lambda\) is very small, the fit is close to unregularised least squares and can overfit the training data, especially with degree 9. Increasing \(\lambda\) penalises large weights and usually smooths the fit. If \(\lambda\) becomes too large, the model underfits. Test RMSE is used to select \(\lambda^\star\); its exact value and the 5% interval must be taken from the program output.

### P3 — Lasso and Ridge

**What are the Lasso optimality (KKT) conditions?**

Let \(r=t-\Phi w\). For a coefficient that is not approximately zero, optimality requires
\[
\Phi_j^T r=\lambda\,\operatorname{sign}(w_j).
\]
For a coefficient that is exactly zero, the condition is
\[
|\Phi_j^T r|\leq\lambda.
\]
The assignment uses the numerical threshold \(|w_j|<2\eta\lambda\) to classify an approximately-zero coefficient. Therefore, report the maximum absolute violation of the first condition for the other coefficients and the maximum ratio \(|\Phi_j^Tr|/\lambda\) for approximately-zero coefficients. The latter should be at most 1 at an optimum; a larger value indicates that the iterate has not closely satisfied the KKT conditions.

**Why does L1 regularisation produce zero coefficients while L2 usually only shrinks them?**

The L1 penalty has corners at zero. When a feature's residual correlation is small enough, the optimum can lie at one of those corners, making its coefficient exactly zero. The L2 penalty is smooth and generally pulls coefficients toward zero without setting them exactly to zero. Thus, Lasso can produce a sparse model, while Ridge typically retains all coefficients.

**Why might the objective fluctuate during subgradient descent?**

The L1 objective is not differentiable at zero, and the implementation uses a constant step size. Subgradient descent is not guaranteed to reduce the objective at every step, so individual values can rise or fall and coefficients can chatter around zero. The assignment therefore asks for a best-so-far objective curve for convergence visualisation, while RMSE, coefficient counts, norms, and fitted curves must be computed from the final iterate after 20,000 updates.

**How do training and test errors change as \(\lambda\) increases beyond \(\lambda^\star\)?**

As regularisation strengthens, the model has less flexibility, so training error generally increases. Test error may initially improve as overfitting is reduced, but after the best value \(\lambda^\star\), stronger regularisation generally causes underfitting and test error increases. Use the RMSE plot to describe the observed trend in your run.

### P4 — Bayesian regression

**Why is the posterior over the weights Gaussian?**

The likelihood is Gaussian in the linear model weights, and the prior over the weights is also Gaussian. Multiplying these distributions and normalising gives a Gaussian posterior \(p(w\mid t)=\mathcal N(m_N,S_N)\), with
\[
S_N^{-1}=\alpha I+\beta\Phi^T\Phi,\qquad
m_N=\beta S_N\Phi^Tt.
\]

**Why does the Bayesian posterior mean match the MAP/Ridge solution?**

The posterior is Gaussian, so its mean and mode are the same point \(m_N\). The MAP estimate solves the Ridge system with \(\lambda=\alpha/\beta\):
\[
w_{\mathrm{MAP}}=(\Phi^T\Phi+\lambda I)^{-1}\Phi^Tt.
\]
Substituting \(\lambda=\alpha/\beta\) into this system gives the same solution as the posterior-mean formula. The Euclidean difference \(\|w_{\mathrm{MAP}}-m_N\|_2\) should therefore be close to numerical precision.

**Why does predictive uncertainty shrink as the training-set size increases?**

More observations provide more information about the polynomial weights, reducing posterior covariance \(S_N\) and therefore reducing the parameter-uncertainty term \(\phi(x)^TS_N\phi(x)\). The predictive variance is
\[
s^2(x)=\underbrace{\beta^{-1}}_{\text{observation noise}}+
\underbrace{\phi(x)^TS_N\phi(x)}_{\text{parameter uncertainty}}.
\]
The parameter-uncertainty term is expected to shrink as \(N\) increases. The observation-noise term \(\beta^{-1}=0.01\) remains because a future observation is still noisy, even if the model parameters were known perfectly. Consequently, the 95% predictive interval approaches the irreducible width \(2(1.96)/\sqrt{\beta}\).

