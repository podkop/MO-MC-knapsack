import mip
import knaps_base
import numpy as np
import time

class knapsack_cbc(knaps_base.knapsack_base):
    _var_types = {
        "continuous": mip.CONTINUOUS, 
        "binary": mip.BINARY,
        "integer": mip.INTEGER
        }
    _last_opt_time = None
    # Maximum duration of optimization process
    _max_time = mip.INF
    
    def _create_model(self, name="", max_time = None, **kwargs):
        self._model = mip.Model(solver_name = "CBC")
        if max_time is not None:
            self._max_time = max_time

## Adding a variable: shape = None => single; otherwise (multi)dimensional
#  If shape is int (incl. 1) or (int,), then create an 1D-vector variable
#  If lb_inf, then not bounded from below (otherwise >= 0)
    def _add_var(self, shape = None, vtype = "continuous", lb_inf = False,
                 lb = 0.0, ub = mip.INF, name = ""):
        if lb_inf:
            lb = -mip.INF
        if shape is None:
            return self._model.add_var(
                name = name,
                lb = lb, ub = ub,
                var_type = self._var_types[vtype]
                )
        if not hasattr(shape,"__len__"):
            shape = [shape]
        newshape = None if len(shape) == 1 else shape[1:]
        return [
            self._add_var(
                shape = newshape, 
                vtype = vtype, 
                lb = lb, ub = ub, 
                name = name
                )
                    for _ in range(shape[0])
                    ]

## Returns value of the given single variable
    def _var_value(self,v):
        return v.x


## Delete an object (variable, constraint) or a list of objects from the model
#  If list cannot be removed, removes recursively, passing exception to leafs
    def _remove(self, ol):
        try:
            self._model.remove(ol)
        except:
            if isinstance(ol, list):
                for oi in ol:
                    self._remove(oi)
            else:
                self._model.remove(ol)

## Updates the model
    def _upd(self):
        pass

## Sum of binary vars in (уфср) row == / <= / >= value(s)
#!todo: Currently implemented only structural constraints
#!todo: add parameters for all/specific rows, constr. type, rhs
    def _add_row_sum_constr(self):
        for vi in self._vars_x:
            self._model.add_constr(mip.xsum(vi) == 1)

## Add the linear constraint with all binary vars, coeffs of matrix shape
    def _add_full_constr(self, coeff_matr, rhs = 0.0, sense = "<="):
        if sense == "<=" or sense == "<":
            return self._model.add_constr(
                mip.xsum( 
                    xi * ci for xxi, cci in zip(self._vars_x, coeff_matr) 
                        for xi, ci in zip(xxi,cci) 
                            ) <= rhs
                )
        elif sense == ">=" or sense == ">":
            return self._model.add_constr(
                mip.xsum( 
                    xi * ci for xxi, cci in zip(self._vars_x, coeff_matr) 
                        for xi, ci in zip(xxi,cci) 
                            ) >= rhs
                )
        elif sense == "==" or sense == "=":
            return self._model.add_constr(
                mip.xsum( 
                    xi * ci for xxi, cci in zip(self._vars_x, coeff_matr) 
                        for xi, ci in zip(xxi,cci) 
                            ) == rhs
                )

## Add the linear constraint with binary vars of j-th column, with a column of coeff    
    def _add_col_constr(self, coeff_col, j, rhs = 0.0, sense = "<="):
        if sense == "<=" or sense == "<":
            return self._model.add_constr(
                    mip.xsum(
                        coeff_col[i] * self._vars_x[i][j] 
                            for i in range(self._m) )
                    <= rhs
                )
        elif sense == ">=" or sense == ">":
            return self._model.add_constr(
                    mip.xsum(
                        coeff_col[i] * self._vars_x[i][j] 
                            for i in range(self._m) )
                    >= rhs
                )
        elif sense == "=" or sense == "==":
            return self._model.add_constr(
                    mip.xsum(
                        coeff_col[i] * self._vars_x[i][j] 
                            for i in range(self._m) )
                    >= rhs
                )
        
## Adds a constraint from givel lists of vars and weights
    def _add_constr_list(self, var_l, w_l, rhs = 0.0, sense = "<="):
        if sense == "<=" or sense == "<":
            return self._model.add_constr(
                mip.xsum( wi*vi for vi, wi in zip(var_l,w_l) ) 
                    <= rhs
                )
        elif sense == ">=" or sense == ">":
            return self._model.add_constr(
                mip.xsum( wi*vi for vi, wi in zip(var_l,w_l) ) 
                    >= rhs
                )
        elif sense == "=" or sense == "==":
            return self._model.add_constr(
                mip.xsum( wi*vi for vi, wi in zip(var_l,w_l) ) 
                    == rhs
                )

## Set upper bound for a given model's variable. None -> ub=inf
    def _set_ub(self, v, ub = None):
        if ub is None:
            ub = mip.INF
        v.ub = ub


## Returns last solution (binary bars): np. matrix or list of np. vectors
    def _sol_x(self):
        if self._q_rectang:
            return np.array(
                [ [xi.x for xi in vi ] for vi in self._vars_x ]
                )
        else:
            return [ [xi.x for xi in vi ] for vi in self._vars_x ]

## Returns True if optimal solution was found, otherwise False
    def _q_opt(self):
        if self._model.status.value == 0:
            return True
        else:
            return False
        
## Returns optimization status
    def _opt_status(self):
        return self._model.status.value

## Returns MIP gap or None
    def _opt_mipgap(self):
        return self._model.gap

## Returns time of last optimization (sec) or None
    def _opt_time(self):
        return self._last_opt_time

## Sets optimization objective, givel lists of vars and weights
    def _setobj_list(self, var_l, w_l):
        self._model.objective = (
            mip.xsum( vi*wi for vi, wi in zip(var_l,w_l) )
            )

## Calls optimizer
    def _optimize(self):
        t = time.time()
        self._model.optimize( max_seconds = self._max_time )
        self._last_opt_time = time.time() - t