import gurobipy as gbp
import knaps_base
import numpy as np

### All variables are of MVar type


class knapsack_gurobi(knaps_base.knapsack_base):
    _var_types = {
        "continuous": gbp.GRB.CONTINUOUS, 
        "binary": gbp.GRB.BINARY, 
        "integer": gbp.GRB.INTEGER
        }
    
    
    def _create_model(self, name="", max_time = None, **kwargs):
        self._model = gbp.Model(name="")
        self._model.params.NumericFocus=3
        if max_time is not None:
            self._model.params.TimeLimit = max_time
        if "mute" in kwargs:
            self._model.params.outputflag = 0

## Adding a variable: shape=None => single; otherwise multidimensional
#  If shape is int (including 1), then create an 1D-vector variable
#  If lb_inf, then not bounded from below (otherwise >= 0)
    def _add_var(self, shape = None, vtype = "continuous", lb_inf = False,
                 lb = 0.0, ub = gbp.GRB.INFINITY, name = ""):
        if lb_inf:
            lb = -gbp.GRB.INFINITY
        if shape is None:
            shape = 1
        return self._model.addMVar(
            shape, 
            lb = lb, ub = ub, 
            vtype = self._var_types[vtype],
            name = name
            )

## Returns value of the given single variable
    def _var_value(self,v):
        return v.x[0]


## Delete an object (variable, constraint) or a list of objects from the model
#  If list cannot be removed, removes recursively, passing exception to leafs
    def _remove(self, ol):
        try:
            self._model.remove(ol)
        except:
            if isinstance(ol, list):
                for oi in ol:
                    self._remove(oi)
            elif isinstance(ol, gbp.MVar):
                self._model.remove(ol.vararr.flatten().tolist())
            else:
                self._model.remove(ol)

## Updates the model
    def _upd(self):
        self._model.update()

## Sum of binary vars in (уфср) row == / <= / >= value(s)
#!todo: Currently implemented only structural constraints
#!todo: add parameters for all/specific rows, constr. type, rhs
    def _add_row_sum_constr(self):
        for vi in self._vars_x:
            self._model.addConstr(vi.sum() == 1)

## Add the linear constraint with all binary vars, coeffs of matrix shape
    def _add_full_constr(self, coeff_matr, rhs = 0.0, sense = "<="):
        if sense == "<=" or sense == "<":
            return self._model.addConstr(
                sum( np.array(ci) @ vi 
                    for vi, ci in zip(self._vars_x, coeff_matr) ) - rhs <= 0
                    )
        elif sense == ">=" or sense == ">":
            return self._model.addConstr(
                sum( np.array(ci) @ vi 
                    for vi, ci in zip(self._vars_x, coeff_matr) ) - rhs >= 0
                    )
        elif sense == "=" or sense == "==":
            return self._model.addConstr(
                sum( np.array(ci) @ vi 
                    for vi, ci in zip(self._vars_x, coeff_matr) ) - rhs == 0
                    )

## Add the linear constraint with binary vars of j-th column, with a column of coeff    
    def _add_col_constr(self, coeff_col, j, rhs = 0.0, sense = "<="):
        if sense == "<=" or sense == "<":
            return self._model.addConstr(
                    coeff_col @ self._vars_x[:,j] - rhs <= 0.0
                )
        elif sense == ">=" or sense == ">":
            return self._model.addConstr(
                    coeff_col @ self._vars_x[:,j] - rhs >= 0.0
                )
        elif sense == "=" or sense == "==":
            return self._model.addConstr(
                    coeff_col @ self._vars_x[:,j] - rhs == 0.0
                )
## Adds a constraint from givel lists of vars and weights
    def _add_constr_list(self, var_l, w_l, rhs = 0.0, sense = "<="):
        if sense == "<=" or sense == "<":
            return self._model.addConstr(
                sum( wi*vi for vi, wi in zip(var_l,w_l) ) <= rhs
                )
        elif sense == ">=" or sense == ">":
            return self._model.addConstr(
                sum( wi*vi for vi, wi in zip(var_l,w_l) ) >= rhs
                )
        elif sense == "=" or sense == "==":
            return self._model.addConstr(
                sum( wi*vi for vi, wi in zip(var_l,w_l) ) == rhs
                )
## Set upper bound for a given model's variable. None -> ub=inf
    def _set_ub(self, v, ub = None):
        if ub is None:
            ub = gbp.GRB.INFINITY
        v.ub = ub


## Returns last solution (binary bars): np. matrix or list of np. vectors
    def _sol_x(self):
        if self._q_rectang:
            return self._vars_x.getAttr("X")
        else:
            return [ vi.getAttr("X") for vi in self._vars_x ]

## Returns True if optimal solution was found, otherwise False
    def _q_opt(self):
        if self._model.Status in [2,9]:
            return True
        else:
            return False
## Returns optimization status
    def _opt_status(self):
        return self._model.Status

## Returns MIP gap or None
    def _opt_mipgap(self):
        try:
            return self._model.MIPGap
        except:
            return None

## Returns time of last optimization (sec) or None
    def _opt_time(self):
        try:
            return self._model.RunTime
        except:
            return None

## Sets optimization objective, givel lists of vars and weights
    def _setobj_list(self, var_l, w_l):
        self._model.setObjective(
            sum( wi*vi for vi, wi in zip(var_l,w_l) )
            )
## Calls optimizer
    def _optimize(self):
        self._model.optimize()