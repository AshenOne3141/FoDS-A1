# Foundations of Data Science

## Assignment 1 --- Code Documentation

This document describes the supplied Python implementation for P1--P4,
following the organisation and requirements of the **CS F320:
Foundations of Data Science Assignment-1** PDF. It explains the modules,
algorithms, inputs, outputs, and how to run the menu-driven CLI.

  -----------------------------------------------------------------------
  Module                  Topic                   Main implementation
  ----------------------- ----------------------- -----------------------
  `p1.py`                 Polynomial regression   Least-squares fit;
                                                  QR-reparameterised
                                                  batch, stochastic, and
                                                  mini-batch gradient
                                                  descent; plots and
                                                  verification.

  `p2.py`                 Ridge regression        Closed-form ridge,
                                                  gradient descent,
                                                  regularisation sweep,
                                                  train/test RMSE and
                                                  fitted curves.

  `p3.py`                 Lasso regression        Legendre basis,
                                                  subgradient descent,
                                                  KKT checks, coefficient
                                                  paths, RMSE sweep,
                                                  ridge comparison.

  `p4.py`                 Bayesian regression     Posterior
                                                  mean/covariance,
                                                  predictive uncertainty,
                                                  MAP/ridge equivalence,
                                                  effect of sample size.

  `cli.py`                Program launcher        Interactive
                                                  command-line menu to
                                                  select and run the
                                                  available tasks in
                                                  P1--P4.
  -----------------------------------------------------------------------

## 1. Project layout and data

The code expects to be run from a project folder containing the Python
files and the supplied CSV datasets. The CLI imports the modules from a
folder named `src` and searches for the CSVs in relative locations.

Recommended layout:

``` text
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
    └── results/
```

-   Each CSV is expected to contain two columns, `x` and `y`, with a
    header row. The code loads numeric rows with NumPy and skips the
    header.
-   Use the fixed provided datasets. The assignment states that data
    should not be generated or modified and that `sin(2πx)` is for
    plotting only.
-   The scripts save selected weight vectors under `data/results`.
    Create this directory before running tasks that call `np.savetxt`.
-   The CLI's P1 loader checks `data/poly_train.csv` and
    `../data/testing_training/poly_train.csv` relative to the project
    root; P2--P4 use `../data/testing_training` paths. Keep the folder
    layout consistent with these relative paths, or adjust the paths in
    `cli.py`.

## 2. Running the program

Install the required packages:

``` bash
pip install numpy matplotlib
```

From the project root, run:

``` bash
python cli.py
```

The main menu lets you select P1, P2, P3, P4, or exit. Each problem has
a submenu for running individual plots/checks or a group of tasks. Use
option `0` to return to the previous menu. Plot windows must be closed
before the program can continue to the next plotting task.

**Important import note:** keep module-level execution limited. The CLI
imports `p1.py`--`p4.py`, so data loading or plotting that should happen
only when a file is run directly should be placed inside an
`if __name__ == "__main__":` block. The currently supplied
`p1.py`--`p3.py` load some CSVs at import time, which means their paths
must already be valid even before a menu option is selected.

## 3. P1 --- Polynomial regression and gradient descent

### Goal and model

P1 fits polynomial models of degrees 1 through 9 to the ten training
samples. For degree (d), the monomial design matrix contains columns (1,
x, x\^2, `\ldots`{=tex}, x\^d). The objective is

\[ E(w) = `\frac{1}{2}`{=tex}\|`\Phi `{=tex}w - t\|\^2. \]

### Methods

-   **Closed form:** builds a Vandermonde design matrix and obtains
    weights with `np.linalg.lstsq`, a stable least-squares solver.
-   **QR reparameterisation:** factors (`\Phi `{=tex}= QR) and optimises
    in (`\alpha`{=tex})-space using (Q), then recovers (w) by solving
    (Rw=`\alpha`{=tex}). This reduces the numerical difficulty of
    degree-9 optimisation.
-   **Batch GD:** updates the parameter vector once per epoch using the
    full gradient (Q\^T(Q`\alpha`{=tex}-t)).
-   **Stochastic GD:** shuffles the samples each epoch and updates from
    one sample at a time. The random generator uses seed `42`.
-   **Mini-batch GD:** shuffles samples and updates using batches of
    size 3.

### CLI options

1.  Plot closed-form fits for degrees 1--9 alongside training points and
    the true sine curve.
2.  Plot loss versus epoch for batch, stochastic, and mini-batch GD.
3.  Compare learning rates `0.01`, `0.5`, and `2.1` over 100 epochs;
    print final weights and loss for each.
4.  Compare GD weight vectors against the closed-form solution using
    absolute and relative Euclidean errors.

### Outputs

Plots include nine fitted curves, a combined log-scale loss plot, and a
learning-rate comparison. Weight vectors are written to:

-   `data/results/p1_weights_closed.csv`
-   `data/results/p1_weights_batch.csv`
-   `data/results/p1_weights_stochastic.csv`
-   `data/results/p1_weights_minibatch.csv`

## 4. P2 --- Ridge regression

### Goal and objective

P2 adds an L2 penalty to polynomial least squares. The assignment's
objective is

\[ `\widetilde `{=tex}E(w) =
`\frac{1}{2}`{=tex}\|`\Phi `{=tex}w-t\|^2+`\frac{\lambda}{2}`{=tex}\|w\|^2.
\]

For degree 9 in the monomial basis, the closed-form solution is obtained
by solving

\[ (`\Phi`{=tex}^T`\Phi`{=tex}+`\lambda `{=tex}I)w=`\Phi`{=tex}^Tt. \]

### Methods and outputs

-   **Closed form:** solves the regularised linear system with
    `np.linalg.solve`.
-   **Gradient descent:** starts at zero and uses the ridge gradient
    (`\Phi`{=tex}\^T(`\Phi `{=tex}w-t)+`\lambda `{=tex}w). The code
    compares its weights with the closed-form weights and reports the
    iteration count and condition numbers.
-   **Regularisation sweep:** evaluates (`\lambda`{=tex}) over
    `np.logspace(-10, 2, 49)`, calculates training and test RMSE, and
    chooses (`\lambda`{=tex}\^`\star`{=tex}) with the lowest test RMSE.
-   **Fitted curves:** compares (`\lambda=10`{=tex}\^{-10}), the
    selected (`\lambda`{=tex}\^`\star`{=tex}), and
    (`\lambda=10`{=tex}\^{-2}) against the training data and
    (`\sin`{=tex}(2`\pi `{=tex}x)).

The CLI provides an RMSE sweep, a closed-form solution at
(`\lambda=0.01`{=tex}), and a combined option for the available P2
plots. The RMSE plot includes `0.1` as the reference noise level. The
code also prints a 5% interval of (`\lambda`{=tex}) values whose test
RMSE is no more than 1.05 times the minimum.

## 5. P3 --- Lasso using subgradient descent

### Basis and objective

P3 uses the degree-9 shifted Legendre basis (P_j(2x-1)), implemented
with `np.polynomial.legendre.legvander(2*x - 1, 9)`. This spans the same
degree-9 polynomial function class as the monomial basis but is
generally better conditioned. The Lasso objective is

\[
E(w)=`\frac{1}{2}`{=tex}\|`\Phi `{=tex}w-t\|\^2+`\lambda`{=tex}\|w\|\_1.
\]

### Methods and outputs

-   **Subgradient descent:** starts from zero and performs 20,000
    updates with step size `0.05` for each (`\lambda`{=tex}) in
    `np.logspace(-4, 0, 21)`.
-   **Convergence and KKT check:** at (`\lambda=0.1`{=tex}), plots the
    objective and best-so-far objective and reports KKT-related
    residual-correlation checks against the provided reference objective
    and coefficient vector.
-   **RMSE sweep:** plots training and test RMSE against
    (`\lambda`{=tex}) and reports the best test-RMSE (`\lambda`{=tex})
    and approximate-zero coefficients.
-   **Coefficient paths:** plots each coefficient versus
    (`\lambda`{=tex}) and reports approximate-zero counts for each
    value.
-   **Ridge comparison:** fits ridge regression in the Legendre basis,
    chooses each model's best (`\lambda`{=tex}) by test RMSE, and
    compares test RMSE, coefficient norm, and coefficient counts below
    `0.01`.
-   **Fitted curves:** overlays the selected Lasso and Ridge fits,
    training data, and true sine curve.

The assignment defines an approximate-zero coefficient by
(\|w_j\|\<2`\eta`{=tex}`\lambda`{=tex}); with (`\eta=0.05`{=tex}), the
code uses this threshold in its sweep and coefficient-count
calculations.

## 6. P4 --- Bayesian polynomial regression

### Model and hyperparameters

P4 uses degree 3 with observation-noise precision (`\beta=100`{=tex})
(noise variance `0.01`) and prior precision (`\alpha=0.05`{=tex}). The
posterior weight precision is

\[ S_N^{-1}=`\alpha `{=tex}I+`\beta`{=tex}`\Phi`{=tex}^T`\Phi`{=tex}, \]

and the code solves for covariance (S_N) and computes posterior mean

\[ m_N=`\beta `{=tex}S_N`\Phi`{=tex}\^Tt. \]

### Predictive distribution

For a point (x), the predictive mean is (m(x)=`\phi`{=tex}(x)\^Tm_N).
The predictive variance is

\[ s^2(x)=`\beta`{=tex}^{-1}+`\phi`{=tex}(x)\^TS_N`\phi`{=tex}(x), \]

which includes both observation noise and parameter uncertainty. The
plot shows the mean and the 95% predictive interval
(m(x)`\pm`{=tex}1.96s(x)).

### CLI options

1.  Predictive mean and 95% interval using `p4_train_N10.csv`.
2.  Compare the Bayesian posterior mean with the MAP/Ridge estimate
    using (`\lambda`{=tex}=`\alpha`{=tex}/`\beta=0.0005`{=tex}); print
    the Euclidean difference between the weight vectors.
3.  Plot 95% predictive interval widths for (N=5,10,20,50), with the
    irreducible noise floor.

The expected trend is that the parameter-uncertainty contribution
decreases as (N) increases, while the observation-noise contribution
remains.

## 7. Function reference

  ----------------------------------------------------------------------------------------------------------------------------------------
  File                    Function / method                                                                        Purpose
  ----------------------- ---------------------------------------------------------------------------------------- -----------------------
  `p1.py`                 `closed_form(train_data, d)`                                                             Least-squares weights
                                                                                                                   for a degree-(d)
                                                                                                                   monomial polynomial.

  `p1.py`                 `gradient_descent.batch / stochastic / mini_batch`                                       QR-reparameterised
                                                                                                                   optimisation variants;
                                                                                                                   return weights and loss
                                                                                                                   history.

  `p1.py`                 `plotting.closed_form_curves / loss_curves / learning_rate_study / verification`         P1 figures and
                                                                                                                   numerical comparisons.

  `p2.py`                 `closed_form(train_data, d, lamda)`                                                      Closed-form ridge
                                                                                                                   weights and objective.

  `p2.py`                 `gradient_descent.batch(train_data, d, lamda, learning_rate)`                            Ridge GD with loss
                                                                                                                   history and
                                                                                                                   conditioning
                                                                                                                   diagnostics.

  `p2.py`                 `plotting.sweep / plot_rmse`                                                             Evaluate the fixed
                                                                                                                   (`\lambda`{=tex}) grid
                                                                                                                   and plot RMSE.

  `p3.py`                 `closed_form_ridge_leg(train_data, d, lamda)`                                            Ridge fit in the
                                                                                                                   Legendre basis.

  `p3.py`                 `gradient_descent.batch(train_data, d, lamda, learning_rate)`                            Lasso subgradient
                                                                                                                   iterations and
                                                                                                                   objective history.

  `p3.py`                 `plotting.verify / sweep / plot_rmse / coefficient_path / compare_ridge / plot_curves`   P3 verification,
                                                                                                                   sweeps, coefficient
                                                                                                                   paths, and comparison
                                                                                                                   plots.

  `p4.py`                 `bayesian_reg.fit / predict`                                                             Compute posterior
                                                                                                                   parameters and
                                                                                                                   predictive
                                                                                                                   mean/standard
                                                                                                                   deviation.

  `p4.py`                 `plotting.plot_predictive / equivalence / plot_width`                                    P4 predictive plot, MAP
                                                                                                                   comparison, and
                                                                                                                   sample-size comparison.
  ----------------------------------------------------------------------------------------------------------------------------------------

## 8. Implementation notes to check before submission

This documentation describes the supplied code, not a claim that every
detail has already been verified against every rubric item. The
following consistency checks are worth addressing before treating the
results as final:

-   **P1 mini-batch averaging:** the current mini-batch update uses the
    sum of sample gradients in each batch. The assignment specifies the
    averaged gradient over the mini-batch; divide the batch gradient by
    the batch size to match that requirement.
-   **P2 regularisation convention:** `p2.py` sets `I[0,0] = 0`, so the
    intercept is not penalised in the solved system. The assignment's
    stated formula uses (`\lambda `{=tex}I) for all coefficients. Also,
    the printed objective includes a penalty on all entries, so it is
    not fully consistent with the solved system.
-   **P3 regularisation convention:** `p3.py` sets `I[0,0] = 0` in the
    subgradient update and ridge system, while some objective and KKT
    calculations apply the penalty to all coefficients. Make the penalty
    convention consistent with the assignment formula
    (`\lambda`{=tex}\|w\|\_1) and its KKT checks, unless your course
    instructions specify otherwise.
-   **P2 degree argument:** `gradient_descent.batch` calls
    `closed_form(train_data, 9, lamda)` regardless of the `d` argument,
    and the relative-error calculation has no explicit zero-denominator
    guard. This is harmless only when `d=9` and the reference norm is
    nonzero.
-   **Paths and output directory:** the CLI expects a particular project
    layout and `data/results` to exist. Verify the relative paths before
    running the menu.

These notes are included to make the documentation transparent and help
align the implementation with the assignment PDF. They do not alter the
submitted source files.
