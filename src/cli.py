import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import p1
    import p2
    import p3
    import p4
except Exception as exc:
    print("Import failed. Put cli.py in the project root and p1.py-p4.py in ./src.")
    print(f"{type(exc).__name__}: {exc}")
    raise SystemExit(1)


def load_data(relative_path):
    path = ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return np.loadtxt(path, delimiter=",", skiprows=1)


def get_p1_data():
    for relative_path in ("data/poly_train.csv", "../data/testing_training/poly_train.csv"):
        path = ROOT / relative_path
        if path.exists():
            return np.loadtxt(path, delimiter=",", skiprows=1)
    raise FileNotFoundError("Could not find poly_train.csv in ./data or ./data/testing_training.")


def get_p2_p3_data():
    return (
        load_data("../data/testing_training/poly_train.csv"),
        load_data("../data/testing_training/poly_test.csv"),
    )


def p1_menu():
    while True:
        print("\nP1 — Polynomial Regression / Gradient Descent")
        print("1. Closed-form curves for degrees 1–9")
        print("2. Batch, SGD, and mini-batch loss curves")
        print("3. Learning-rate study")
        print("4. Verify GD against closed form")
        print("5. Run all P1 tasks")
        print("0. Back")
        choice = input("Choose an option: ").strip()
        try:
            train = get_p1_data()
            if choice == "1":
                p1.plotting.closed_form_curves(train)
            elif choice == "2":
                p1.plotting.loss_curves(train, 9, 0.1, 2000)
            elif choice == "3":
                p1.plotting.learning_rate_study(train, 9)
            elif choice == "4":
                p1.plotting.verification(train, 9, 0.1, 2000)
            elif choice == "5":
                p1.plotting.closed_form_curves(train)
                p1.plotting.loss_curves(train, 9, 0.1, 2000)
                p1.plotting.learning_rate_study(train, 9)
                p1.plotting.verification(train, 9, 0.1, 2000)
            elif choice == "0":
                return
            else:
                print("Invalid choice.")
        except Exception as exc:
            print(f"P1 error: {type(exc).__name__}: {exc}")


def p2_menu():
    while True:
        print("\nP2 — Ridge Regression")
        print("1. RMSE sweep")
        print("2. Closed-form Ridge at lambda = 0.01")
        print("3. Run all available P2 plots")
        print("0. Back")
        choice = input("Choose an option: ").strip()
        try:
            train, test = get_p2_p3_data()
            if choice == "1":
                p2.plotting.plot_rmse(train, test)
            elif choice == "2":
                p2.closed_form(train, 9, 0.01)
            elif choice == "3":
                p2.plotting.plot_rmse(train, test)
                if hasattr(p2.plotting, "plot_curves"):
                    p2.plotting.plot_curves(train, test)
            elif choice == "0":
                return
            else:
                print("Invalid choice.")
        except Exception as exc:
            print(f"P2 error: {type(exc).__name__}: {exc}")


def p3_menu():
    while True:
        print("\nP3 — Lasso / L1-Regularised Regression")
        print("1. Convergence and KKT verification at lambda = 0.1")
        print("2. Training/test RMSE sweep")
        print("3. Coefficient path and near-zero counts")
        print("4. Lasso vs Ridge comparison")
        print("5. Fitted-curve comparison")
        print("6. Run all P3 tasks")
        print("0. Back")
        choice = input("Choose an option: ").strip()
        try:
            train, test = get_p2_p3_data()
            if choice == "1":
                p3.plotting.verify(train, 0.1, 0.05)
            elif choice == "2":
                p3.plotting.plot_rmse(train, test)
            elif choice == "3":
                p3.plotting.coefficient_path(train, test)
            elif choice == "4":
                p3.plotting.compare_ridge(train, test)
            elif choice == "5":
                p3.plotting.plot_curves(train, test)
            elif choice == "6":
                p3.plotting.verify(train, 0.1, 0.05)
                p3.plotting.plot_rmse(train, test)
                p3.plotting.coefficient_path(train, test)
                p3.plotting.compare_ridge(train, test)
                p3.plotting.plot_curves(train, test)
            elif choice == "0":
                return
            else:
                print("Invalid choice.")
        except Exception as exc:
            print(f"P3 error: {type(exc).__name__}: {exc}")


def p4_menu():
    while True:
        print("\nP4 — Bayesian Regression")
        print("1. Predictive mean and 95% interval (N = 10)")
        print("2. Bayesian mean vs MAP/Ridge equivalence")
        print("3. Predictive interval width for N = 5, 10, 20, 50")
        print("4. Run all P4 tasks")
        print("0. Back")
        choice = input("Choose an option: ").strip()
        try:
            train10 = load_data("../data/testing_training/p4_train_N10.csv")
            if choice == "1":
                p4.plotting.plot_predictive(train10, 100.0, 0.05, 3)
            elif choice == "2":
                p4.plotting.equivalence(train10, 100.0, 0.05, 3)
            elif choice == "3":
                p4.plotting.plot_width(100.0, 0.05, 3)
            elif choice == "4":
                p4.plotting.plot_predictive(train10, 100.0, 0.05, 3)
                p4.plotting.equivalence(train10, 100.0, 0.05, 3)
                p4.plotting.plot_width(100.0, 0.05, 3)
            elif choice == "0":
                return
            else:
                print("Invalid choice.")
        except Exception as exc:
            print(f"P4 error: {type(exc).__name__}: {exc}")


def main():
    while True:
        print("\n========================================")
        print("FoDS Assignment Runner")
        print("========================================")
        print("1. P1 — Polynomial Regression and GD")
        print("2. P2 — Ridge Regression")
        print("3. P3 — Lasso Regression")
        print("4. P4 — Bayesian Regression")
        print("0. Exit")
        choice = input("Choose a problem: ").strip()
        if choice == "1":
            p1_menu()
        elif choice == "2":
            p2_menu()
        elif choice == "3":
            p3_menu()
        elif choice == "4":
            p4_menu()
        elif choice == "0":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Enter 0–4.")


if __name__ == "__main__":
    main()
