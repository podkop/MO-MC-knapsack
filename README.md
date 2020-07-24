# Multiobjective multiple-choice knapsack problems

This repository implements functions to create problems from data and solve them with selected Mixed Integer Linear Programming (MILP) solvers from Python 3. The intention is to handle large problems efficiently, and also to enable comparing different solvers without excessive programming.

**The class of problems** is multiple-choice knapsack problems (see e.g. [[1]](https://doi.org/10.1007/978-3-540-24777-7_11)) formulated with multiple objectives (see e.g. [[2]](https://doi.org/10.1007/978-1-4615-5563-6), [[3]](https://doi.org/10.1007/978-3-319-32756-3)). Single-objective problems can also be handled.

**Solving the multiobjective problem** is understood in terms of solving a scalarized problem. Scalarizations include weighted sum, epsilon-constraint, and achievement scalarizing function (for reference point methods). Calculating the ranges of Pareto front and handling degenerate objective functions is also implemented.

**The problem instance** is created as an object of a class which is specific to the selected MILP solver. The problem can be later modified and solved using this solver. The class interface does not depend on the selected solver, since all the classes are derived from an abstract base class. 

**A MILP solver is called** via the respective Python API. Currently the supported solvers are the following:

| Solver | Python API |
|---|---|
| [CBC (Coin OR)](https://en.wikipedia.org/wiki/COIN-OR#CBC) | [Python-MIP](https://python-mip.com/)
| [Gurobi](https://en.wikipedia.org/wiki/Gurobi)| [GurobiPy](https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html) |

**This repository is structured as follows.** The module knaps_base.py implements the abstract base class, which includes the user interface for creating, modifying the problem and formulating different scalarizations. For each of the supported solvers, a separate module represents the derived class, which adds functions to work with this solver. In order to use this library, import the specific module corresponding to the selected solver. In order to understand the user interface and inner workings, read the base module.
