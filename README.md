# Multiobjective multiple-choice knapsack problems

This repository implements functions to create problems from data and solve them with selected Mixed Integer Linear Programming (MILP) solvers from Python 3. The intention is to handle large problems efficiently, and also to enable comparing different solvers without excessive programming.

**The class of problems** is multiple-choice knapsack problems (see e.g. [[1]](https://doi.org/10.1007/978-3-540-24777-7_11)) formulated with multiple objectives (see e.g. [[2]](https://doi.org/10.1007/978-1-4615-5563-6), [[3]](https://doi.org/10.1007/978-3-319-32756-3)). *Single-objective* problems can also be handled.

**Solving the multiobjective problem** is understood in terms of solving a scalarized problem. *Implemented scalarizations*: weighted sum, epsilon-constraint, and an achievement scalarizing function (ASF) used in reference point methods. Calculating the ranges of Pareto front and handling degenerate objective functions is also included.

**A MILP solver is called** via the respective Python API. Currently the supported solvers are the following:

| Solver | Python API |
|---|---|
| [CBC (Coin OR)](https://en.wikipedia.org/wiki/COIN-OR#CBC) | [Python-MIP](https://python-mip.com/)
| [Gurobi](https://en.wikipedia.org/wiki/Gurobi)| [GurobiPy](https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html) |

# Documentation
This package is still under construction, the deficiencies are described in the last section. The package is provided "as is", the author disclaims all warranties.

## The structure of the package

- `knaps_base.py` is the *general module* containing the abstract base class (ABC), which implements the functionality of creating, modifying and solving the problems. 
- `knaps_cbc.py`, `knaps_cbc.py` are *specific modules* corresponding to the supported solvers. Each specific module represents the class derived from ABC, which connects the functionality from `knaps_base.py` with the corresponding solver.

To ***install*** the package, copy all .py files from the main directory to a folder on your computer, and make sure this folder is [accessible](https://stackoverflow.com/questions/17806673/where-shall-i-put-my-self-written-python-packages/17811151#17811151) from your Python environment. To ***use*** the package, import specific modules corresponding to your selected solvers. To ***understand*** the package, read the general module.

There is [**an example**](https://github.com/podkop/MO-MC-knapsack/tree/master/Examples/Forestry) illustrating work with the package.

## Features

Objective functions (objectives) have their names; an objective function vector is represented as dictionary `{name: value, ...}`. If you do not set a name for an objective, it is assigned with the subsequent integer starting from 0.

When solving a problem, you can choose to optimize a subset of objectives or only one objective.

All objective functions are of the minimization type. When creating the problem, each maximization objective has to be converted to minimization by multiplying coefficients by -1. For convenience, the function `obj2out` can be used to transform selected objective function values back to maximization for output.

The binary variables are represented in the form of a 2D array: each row contain variables from one class. The array is not necessary rectangular (classes may have different sizes). 

In order to avoid numerical issues, by default all problem coefficients are normalized. This does not affect the returned values of objective functions: they are automatically converted back to the original scale before output.

The degeneracy of objectives is detected and taken into account when deriving solutions with ASF.

## Usage

***This description is incomplete, for more details read `knaps_base.py`***

Let a solver-specific module `knaps_slv` be imported, where `slv` corresponds to the selected solver (`cbc` or `gurobi`).

To **create a problem instance**, first initialize the problem object: 
```
Problem = knaps_slv.knapsack_slv(
   var_shape, # shape of the array of binary variables: a tuple (m,n) for rectangular, or a list of row lengths (ints) for non-rectangular
   obj2out = copy.copy, # a function to transform the dictionary of objective function values before output if needed
   name = "", # name of the problem, not used
   normalize = True,  # if problem coefficients should be automatically normalized
   max_time = None,  # optimization time limit, seconds
  )
  ```
In addition, you can use the argument `mute = True` to tell Gurobi not to print optimization log. Note: the structural constraints (for each class, sum of variables = 1) are created automatically.

Then, you have to **add objective functions** one-by-one:
```
Problem.add_obj(
  coeffs, # array of coefficients of the same structure as variable array, given as list of lists, numpy array, or list of numpy arrays
  name = None, # optionally, name of the objective function
  u_bound = None # optionally, the upper bound of the objective function
  )
```

You can add **constraints** as follows:
- set an upper bound on an objective function when adding this function, or later when solving the problem;
- add a "less or equal" constraint involving all the variables:
```
add_constr(
  coeffs, # array of coefficients, same as in add_obj
  rhs = 0.0, # optionally, right hand side
  name = None # optionally, the name of the constraint
  )
```
- add one or several constraints involving individual column(s) of the variable array:
```
add_col_constrs(
  coeffs, # coefficients as an array of one or several columns; one column is automatically populated according to col_ids
  rhs = 0.0, # right hand side as a number or list (for all columns); also populated if needed
  sense = "<=", # type of the constraint(s), "less or equal" by default
  col_ids = None # list of column(s) to create constraint(s) for; None means all the columns
  )
```

It is useful to calculate **ranges of objective function values** for the Pareto optimal set: `Problem.eff_range()`. This is done by deriving the ideal objective vector, and estimating the nadir objective vector from the payoff matrix. The ideal point is often used as the reference point; the nadir-ideal ranges are often used to scale weights, which are initially provided independently on the scales of different objective function values.

Function `eff_range` does not return anything. It create the following attributes (of `dict` type) which can be utilized later:
- `_ideal`, `_nadir` - ideal and nadir points (objective vectors);
- `_scale` - the vector of multipliers for scaling weights, if the latter are given without taking into account scales of objective function values.
- `out_ideal`, `out_nadir` - ideal and nadir points transformed for output.

To solve the problem, use any of the functions described below. Weights of objectives can be provided as a dictionary `{name: weight}` or a list of weights. If weights are not given, they are all set to 1. Not necessary all the objectives are involved in optimization - if weights are given as dictionary where some objectives are missing, they are not involved. To solve single-objective problems, use `solve_lin` or `solve_eps` with the chosen objective.

Linear, or weighted sum scalarization
```
solve_lin(
  w = None, # weights
  q_scale = True, # whether weights should be scaled
  obj_bounds = None, # enforce upper bounds on chosen objectives
  )
```

Epsilon constraint scalarization (not implemented properly - the solution may be weakly efficient):
```
solve_eps(
  obj_name, # name of the objective to minimize
  others = {} # upper bounds on other objectives
  )
```

Achievement scalarizing function. The augmentation coefficient is defined by the attribute `_rho = 0.0001`.
```
solve_asf(
  ref = None, # the objective vector (dict) serving as the reference point; the ideal point is used by default
  w = None, # weights
  q_scale = True, # whether weights should be scaled
  obj_bounds = None, # enforce upper bounds on chosen objectives
  )
``` 

# Deficiencies of this package

The author is not a professional software developer and barely follows good practices of programming.

The package has been tested only with few use cases of author's interest.

Many features included in the architecture are not (yet) implemented, e.g. modification of objectives or constraints, proper solution of epsilon-constraint problems and proper derivation of the payoff matrix, control of treatment of degenerate objectives.
