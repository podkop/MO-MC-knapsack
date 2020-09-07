The problem example is implemented as a [Jupyter notebook](https://jupyter.org/) in the file `frs_main.ipynb`. It uses both CBC and Gurobi solvers. If one of the solvers is not installed on your computer please edit the first cell accordingly.

The rest of the files are not used, and provided for information. The example problem data were randomly generated in `frs_random.xlsx` and using script `frs_random2code.py`, were converted into Python code saved in `frs_input.py`. This code was pasted in a cell of the notebook. The purpose is to avoid using additional packages in the notebook for reading external data.

## Problem
It is a simplification of the problem of multiobjective forest management described in [Peura et al (2016)](https://doi.org/10.14214/sf.1672). The data were generated randomly, without any connection with real world. Proportions between values very roughly reflect the meaning of different indicators.

A forest landscape is divided into 20 forest stands. One of three forest management regimes can be selected to each of the stands independently. The forest management plan is the assignment of a management regime to each stand. The long-term outcome is estimated in terms of three land use characteristics playing the role of maximization objectives. The value of each objective for the forest landscape is the sum of contributions from all forest stands. In its turn, the amount of contribution for each stand is determined based on the selected management regime.

Rows of the binary variable array (classes) are forest stands, columns are management regimes, x(i,j) = 1 iff regime j is selected for stand i. For each objective, the coefficient c(i,j) indicates what is the contribution of stand i to this objective function value, if regime j is selected for this stand.
