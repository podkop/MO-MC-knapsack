from abc import ABC
import numpy as np
import copy

# If all keys are ints 0...k-1, returns k, else False
# Usage: check if objective names are [0,1,...,k-1].
# Then objective values information can be represented as a vector
def numer_keys(d):
    k = len(d)
    int_keys = sorted([i for i in d if type(i) == int])
    if int_keys == list(range(k)):
        return k
    return False

# If dict keys (objective functions names) are ints 0...k-1, 
# return numpy objective vector, otherwise numpy []
# Usage: represent objective values information as a vector if possible
def obj_dict2vect(d):
    k = numer_keys(d)
    if k:
        return np.array([d[ki] for ki in range(k)])
    return np.array([])

# Given a numpy array of coefficients, calculate the factor (multiplicator)
# for normalization of coefficients to avoid big range
def normalize_factor(coefs):
    return 1 / (
        max( coefs.max(), -coefs.min() )
        )

# Single objective - by linear form, multi-objective - by dummy constraints
# - all obj. coeffs are given as minimization; internally normalized if needed
# - input (scalariz. params) and output (obj. vectors, ranges) are normalized
# - when saving or copyting between models, range information is re-normalized
#   "vector"_out - factor out normalization, then transform to meaningful
#   obj2out - function (model's objectives dict without normalization) 
#                -> (meaningful objectives dict)
#   var_shape: tuple (nr. objects, nr. choices) or [for each obj: nr. choices]
#   payoff_status: True is range was calculated, "proper" if using eps-constraint
#   normalize: normalize all model coefficients to avoid issues of big ranges
#       #!todo: currently implemented only for rectangular problem type
#       #!todo: when parsing weights as list, order of objectives is different in solve_lin  
class knapsack_base(ABC):
    _var_types = { # Keywords naming variable types
        "continuous": None, 
        "binary": None, 
        "integer": None
        }
    # If (nadir-ideal)/(scale of objective) <= _nadir_ideal_tol, then degenerate
    _nadir_ideal_tol = 10**-6 
    # Scaling coeff. for degenerate objectives (relative to max of other scales)
    _max_scale_mult = 1000
    # Default value of obj. value (%) for degenerate objectives
    _degen_prc = 100
    # Constant for proper efficiency
    _rho = 0.0001
    def __init__(self, 
                 var_shape, 
                 obj2out = copy.copy, 
                 name = "",
                 normalize = True,
                 max_time = None, # optimization time limit
                 **kwargs
                 ):
        self._shape_x = var_shape
        self._obj2out = obj2out
        self._normalize = normalize
        # If same nr. of choices for each object
        self._q_rectang = True if type(var_shape) == tuple else False
        # nr. of objects
        self._m = var_shape[0] if self._q_rectang else len(var_shape)
        # (max) nr. of choices
        self._n = var_shape[1] if self._q_rectang else max(var_shape)
        # Init. payoff status
        self._payoff_status = False
        ## Initialize storage of solver's model elements
        # Constraints by names
        self._constrs = {} 
        # "var": dummy variable, "constr": dummy constr (var >= lin. expres.)
        self._mobjs = {}
        # Auto-generate constraint's or objective's key if name is not given
        self._next_constr_key = self._next_obj_key = 0
        # Dummy variable for ASF
        self._dummy_asf = None
        ## initialize solver's model
        self._create_model(name, max_time = max_time, **kwargs)
        if self._q_rectang:
            self._vars_x = self._add_var(var_shape, vtype = "binary")
        else:
            self._vars_x = [
                self._add_var(int(ni), vtype = "binary")
                    for ni in var_shape
            ]
        self._upd()
        self._add_row_sum_constr()
        self._upd()

### Solver's interface functions    
    def _create_model(self, name="", max_time = None, **kwargs):
        pass
## Adding a variable: shape=None => single; otherwise multidimensional
#  If shape is int (including 1), then create an 1D-vector variable
#  If lb_inf, then not bounded from below (otherwise >= 0)
    def _add_var( self, shape = None, vtype = "continuous",lb_inf = False,
                  lb = 0.0, ub = float("inf"), name = "" ):
        pass

## Returns value of the given single variable
    def _var_value(self,v):
        pass

## Delete an object (variable, constraint) or a list of objects from the model
    def _remove(self, ol):
        pass
## Updates the model at solver's side
    def _upd(self):
        pass
## Sum of binary vars in (each) row == / <= / >= value(s)
#!todo: Currently implemented only structural constraints
#!todo: add parameters for all/specific rows, constr. type, rhs
    def _add_row_sum_constr(self):
        pass
## Add the linear constraint with all binary vars, coeffs of matrix shape
    def _add_full_constr(self, coeff_matr, rhs = 0.0, sense = "<="):
        pass
## Add the linear constraint with binary vars of j-th column, with a column of coeff    
    def _add_col_constr(self, coeff_col, j, rhs, sense = "<="):
        pass
## Adds a constraint from givel lists of vars and weights
    def _add_constr_list(self, var_l, w_l, rhs = 0.0, sense = "<="):
        pass
## Set upper bound for a given model's variable. None -> ub=inf
    def _set_ub(self, v, ub = None):
        pass
## Returns last solution (binary bars): np. matrix or list of np. vectors
    def _sol_x(self):
        pass
## Returns True if optimal/approximate solution was found, otherwise False
    def _q_opt(self):
        pass
## Returns optimization status
    def _opt_status(self):
        pass
## Returns MIP gap or None
    def _opt_mipgap(self):
        pass
## Returns time of last optimization (sec) or None
    def _opt_time(self):
        pass
## Sets optimization objective, givel lists of vars and weights
    def _setobj_list(self, var_l, w_l):
        pass
## Calls optimizer
    def _optimize(self):
        pass

## Utilities
    # Convert input to a list of names of objectives
    def _parse_names(self, names = None):
        if names is None:
            return list(self._mobjs.keys())
        elif type(names) == int:
            return list(range(names))
        num_k = numer_keys(names)
        if num_k:
            return list(range(num_k))
        return names
    # Given an objective (sub-)vector as dict, returns it without normalization
    def _factor_out(self, obj):
        return {
            ki: vi / self._mobjs[ki]["factor"] 
                for ki, vi in obj.items()
                }
    # Transform coefficients array to an acceptable format
    def _parse_coeffs(self,c):
        if self._q_rectang:
            if not isinstance(c, np.ndarray):
                return np.array(c)
            else:
                return c
        if not isinstance(c[0], np.ndarray):
            return [np.array(vi) for vi in c]
        else:
            return c
    # Given variables array, returns the list of assignments
    def bin2int(self,x):
        return [
            np.where(vi > 0.999)[0][0] for vi in x
            
            ]
        
## Main functions
    def add_constr(self, coeffs, rhs = 0.0, name = None):
        coeffs = self._parse_coeffs(coeffs)
        if name is None:
            name = self._next_constr_key
            self._next_constr_key += 1
        if name in self._constrs:
            print(f"*** Re-writing constraint \"{name}\" ***")
            self._del_constr(name)
        self._constrs[name] = self._add_full_constr( coeffs, rhs, "<=" )
        
    def _del_constr(self, name):
            self._remove(self._constrs[name])
            self._upd()

    ## Coeffs can be 1 column, then it is populated; rhs can be value o [value]
    #  col_ids = list of column nrs to enforce constraints on (None = to all columns)
    def add_col_constrs(self, coeffs, rhs = 0.0, sense = "<=", col_ids = None):
        if not isinstance(coeffs, np.ndarray):
            coeffs = np.array(coeffs)
        # col_ids = list of columns of vars to enforce constraints to
        if col_ids is None:
            col_ids = list(range(self._n))
        n_coef_cols = coeffs.shape[1]
        # parsing coefficients, coeff_ids controls (coeff. column -> var. column)
        coeff_ids = [0 for _ in col_ids] if n_coef_cols == 1 \
            else list(range(len(col_ids)))
        # parsing rhs, r_ids controls (r.h.s. value -> var. column)
        if not hasattr(rhs,"__len__"):
            rhs = [rhs]
        r_ids = [0 for _ in col_ids] if len(rhs) == 1 \
            else list(range(len(col_ids)))
        # normalization if needed
        if self._normalize and self._q_rectang:
            factors = [
                normalize_factor(coeffs[:,j]) 
                    for j in range(n_coef_cols)
                       ]
        else:
            factors = [1.0 for _ in range(n_coef_cols)]
        self._col_constrs = [
            self._add_col_constr(
                coeffs[:,coeff_ids[ij]] * factors[coeff_ids[ij]], 
                j, 
                rhs[r_ids[ij]] * factors[coeff_ids[ij]], 
                sense
                )
                    for ij,j in enumerate(col_ids)
            ]

    def add_obj(self, coeffs, name = None, u_bound = None,
                save_coeffs = True):
        coeffs = self._parse_coeffs(coeffs)
        self._payoff_status = False # reset range information
        # Setting name, checking for duplicate
        if name is None:
            name = self._next_obj_key
            self._next_obj_key += 1
        if name in self._mobjs:
            print(f"*** Re-writing objective \"{name}\" ***")
            self.del_obj(name)
        # Init the structure stoing the objective function
        o = self._mobjs[name] = {}
        # Normalization factor if needed
        if self._normalize and self._q_rectang:
            o["factor"] = normalize_factor(coeffs)
            coeffs *= o["factor"]
        else:
            o["factor"] = 1.0
        if save_coeffs:
            o["coeffs"] = coeffs
        # Additional args when creating the dummy var
        var_args = {"lb_inf": True}
        o["bound"] = u_bound
        if u_bound is not None:
            var_args["ub"] = u_bound * o["factor"]
        o["var"] = self._add_var(**var_args)
        self._upd()
        o["constr"] = name
        self.add_constr(coeffs, o["var"], name = name)
        o["degen"] = False # if the objective is degenerated
        
    def del_obj(self,name = None):
        if name is None:
            for ni in self._mobjs:
                self.del_obj(ni)
            self._next_obj_key = 0
            return
        self._remove(self._mobjs[name]["var"])
        self._del_constr(self._mobjs[name]["constr"])
        del(self._mobjs[name])

    ## Upper bounds on objectives are initially set and sab=ved in obj["bound"]
    #  Can be temporarily changed and then reset to initial
    
    ## Bounds are given as a vector, or as {names:bounds}. None -> inf
    #  They are stored not normalized
    def _obj_bounds(self, b):
        if not isinstance(b,dict):
            b = {i: b[i] for i in sorted(self._mobjs.keys())}
        for ki, vi in b.items():
            self._set_ub(self._mobjs[ki]["var"], vi * self._mobjs[ki]["factor"])
            self._mobjs[ki]["bound"] = vi
        self._upd()

    def _reset_obj_bounds(self, names = None):
        if names is None:
            names = self._mobjs.keys()
        for ki in names:
            self._set_ub(
                self._mobjs[ki]["var"], 
                self._mobjs[ki]["bound"] * self._mobjs[ki]["factor"]
                )
        self._upd()
    
    def _sol_y(self, coeffs, x = None):
        coeffs = self._parse_coeffs(coeffs)
        if x is None:
            x = self._sol_x()
        return np.dot(np.concatenate(coeffs), np.concatenate(x))
    
    # degen_treat decides how to treat degenerate objectives:
    #    "weight": include in scalar. obj. with corresponding weight
    #    "bound": does not include in scalar. obj. but set upper bound to nadir
    def solve_lin(self, w = None, q_scale = True, obj_bounds = None,
                  degen_treat = "weight"):
        # Parsing w to produce {obj_name: weight}
        if w is None:
            w = {ni: 1 for ni in self._mobjs}
        elif not (isinstance(w,dict)):
            w = {i: w[i] for i in sorted(self._mobjs.keys())}
        else:
            w = w.copy()
        # Init objective bounds dictionary, check degenerate objectives
        obj_bounds = {} if obj_bounds is None else obj_bounds.copy()
        if degen_treat == "bound":
            for ni in list(w):
                if self._mobjs[ni]["degen"]:
                    obj_bounds[ni] = self._nadir[ni] / self._mobjs[ni]["factor"]
                    del(w[ni])
        if q_scale:
            w={ ki: wi*self._scale[ki] for ki, wi in w.items() }                    
        self._last_scalariz = {
            "type": "linear", "w": w, "obj_bounds": obj_bounds }
        # create scalar obj
        self._setobj_list(zip(
           *[
                [ self._mobjs[ni]["var"], w[ni] ] for ni in w
            ] 
            ) )
        self._obj_bounds(obj_bounds)
        self._optimize()
        if len(obj_bounds) > 0:
            self._reset_obj_bounds()
        return self._result()

    def solve_asf(self, ref = None, w = None, q_scale = True, 
                  obj_bounds = None, degen_treat = "bound"):
        # Parsing ref to produce {obj_name: asp. level}; ref is not modified
        if ref is None:
            ref = self._ideal
        elif not isinstance(ref,dict):
            ref = {ni: ri for ni,ri in zip( sorted(self._mobjs.keys()), ref )
                    }
        # Parsing w to produce {obj_name: weight}
        if w is None:
            w = {ni: 1 for ni in ref}
        # If w is a vector, elements' order correspond to ref.keys order
        elif not (isinstance(w,dict)): 
            w = {ni: wi for ni,wi in zip(ref.keys(),w) }
        else:
            w = w.copy()
        if q_scale:
            w={ ki: wi*self._scale[ki] for ki, wi in w.items() }
        # Init objective bounds dictionary, check degenerate objectives
        temp_obj_bounds = {} if obj_bounds is None else obj_bounds.copy()
        if degen_treat == "bound":
            for ni in list(w):
                if self._mobjs[ni]["degen"]:
                    temp_obj_bounds[ni] = self._nadir[ni] / self._mobjs[ni]["factor"]
                    del(w[ni])
        self._last_scalariz = {
            "type": "asf", "ref": ref, "w": w, "obj_bounds": temp_obj_bounds }
        # create scalar obj
        # if self._dummy_asf is not None:
        #     self._remove(self._dummy_asf)
        #     self._upd()
        if self._dummy_asf is None:
            self._dummy_asf = self._add_var(lb_inf=True)
            self._upd()
        dummy_constrs = [
            self._add_constr_list(
                [self._dummy_asf, self._mobjs[ki]["var"]],
                [-1, wi], 
                wi*ref[ki]
                )
                for ki, wi in w.items()
            ]
        self._setobj_list( 
            [self._dummy_asf] + [self._mobjs[ki]["var"] for ki in w], 
            [1.0] + [vi * self._rho for ki, vi in w.items()] 
            )
        self._obj_bounds(temp_obj_bounds)
        self._optimize()
        if len(temp_obj_bounds)>0:
            self._reset_obj_bounds()
        self._remove(dummy_constrs)
        return self._result()

    # Eps. constr, others = {obj. name: upper bound}
    def solve_eps(self, obj_name, others = {}):
        self._last_scalariz = {
            "type": "epsilon", "main": obj_name, "others": others, 
            "obj_bounds": {}
            }
        c_eps = [ self._add_constr_list(
            [self._mobjs[ni]["var"]], [1.0], epsi
            )
                for ni, epsi in others.items() 
                ]
        self._setobj_list([self._mobjs[obj_name]["var"]],[1.0])
        self._optimize()
        self._remove(c_eps)
        return self._result()
        
    ## Calculate vector for scaling weights,
    #  check for degeneracy
    def _calc_scale(self, ideal, nadir):
        self._scale = { 
            ni: 1/(nadir[ni]-ideal[ni]) for ni in ideal
                if nadir[ni] > ideal[ni] and
                (nadir[ni]-ideal[ni]) > 
                    max(abs(nadir[ni]),abs(ideal[ni]))*self._nadir_ideal_tol
            }
        # scale for degenerate objectives
        max_scale = max( [1] + list(self._scale.values()) ) * \
            len(ideal) * self._max_scale_mult
        for ni in self._range_names:
            if ni not in self._scale:
                self._scale[ni] = max_scale
                self._mobjs[ni]["degen"] = True
        
    #  q_proper -> use sequential e-constraint 
    def eff_range(self, names = None, q_proper = False):
        self._range_names = self._parse_names(names)
        self._k_range = len(self._range_names)
        self._payoff_approx = []
        self._payoff = []
        #!del? self._asf_names = [] # names of objectives participating in ASF
        #!del? self._asf_k = 0
        ## calculating approximate payoff matrix
        for ni in self._range_names:
            # print("!\nSolving objective",ni,"\n!")
            self.solve_eps(ni)
            self._payoff_approx.append([
                self._sol_y(self._mobjs[ni]["coeffs"])
                    for ni in self._range_names
                ])
        self._ideal_v = np.array(
            [self._payoff_approx[i][i] for i in range(self._k_range)] )
        self._nadir_v = np.max(self._payoff_approx,axis=0)
        self._ideal, self._nadir = [
            { ni: yi for ni, yi in zip(self._range_names, vi) }
                for vi in [self._ideal_v, self._nadir_v]
            ]
        if q_proper: 
            # calculate payoff using eps-constr
            # update nadir
            pass
        else:
            self._payoff = self._payoff_approx.copy()
        self._calc_scale(self._ideal, self._nadir)
        # ideal and nadir in output format
        self.out_ideal = self._obj2out( self._factor_out(self._ideal) )
        self.out_nadir = self._obj2out( self._factor_out(self._nadir) )
        self._payoff_status = "proper" if q_proper else True
        
    ## 2 functions for copying Efficient range between models
    #  taking into account normalization
    def get_range(self):
        dct = {}
        for attri in ["_ideal", "_nadir", "out_ideal", "out_nadir",
                      "_range_names", "_k_range", "_scale",
                      "_payoff_status", "_payoff", "_payoff_approx"]:
            vi = getattr(self,attri)
            if hasattr(vi,"copy"):
                dct[attri] = vi.copy()
            else:
                dct[attri] = vi
        dct["degen_list"] = [
            ni for ni, oi in self._mobjs.items() if oi["degen"]
            ]
        dct["factors"] = {
            ki: self._mobjs[ki]["factor"] for ki in self._range_names
            }
        return dct
    
    def set_range(self,dct):
        for attri in ["out_ideal", "out_nadir",
                      "_range_names", "_k_range",
                      "_payoff_status"]:
            di = dct[attri]
            if hasattr(di,"copy"):
                setattr(self,attri,di.copy())
            else:
                setattr(self,attri,di)
        # Re-normalizing objective values
        for attri in ["_ideal", "_nadir"]:
            di = dct[attri]
            setattr(self, attri, {
                ki: vi / dct["factors"][ki] * self._mobjs[ki]["factor"]
                    for ki, vi in di.items()
                    } )
        self._scale = {
            ki: vi * dct["factors"][ki] / self._mobjs[ki]["factor"]
                for ki, vi in dct["_scale"].items()
                }
        for attri in ["_payoff", "_payoff_approx"]:
            di = dct[attri]
            setattr(self, attri, [
                [
                rowi[i] / dct["factors"][ki] * self._mobjs[ki]["factor"]
                    for i, ki in enumerate(self._range_names)
                    ] for rowi in di
                ] )
        for ni, oi in self._mobjs.items():
            if ni in dct["degen_list"]:
                oi["degen"] = True
            else:
                oi["degen"] = False
            

    # What information to return if optimization failed
    def _failed_result(self):
        return {
             "status": self._opt_status(),
             "mipgap": self._opt_mipgap(),
             "time": self._opt_time()
             }

    # Returns result of last optimization with used objectives names,
    # or result for given solution with all objective names
    def _result(self, x_ = None):
        x = self._sol_x() if x_ is None else x_
        ## obj. names and scalarizing f. value (no obj2out transformation)
        # In case solution is given
        if x_ is not None:
            names = self._mobjs.keys()
            scal_val = None
        # In case solution from last optimization is needed
        elif self._last_scalariz["type"] == "linear":
            if not self._q_opt:
                return self._failed_result()
            names = list(
                set(self._last_scalariz["w"]) | 
                set(self._last_scalariz["obj_bounds"])
                )
            scal_val = sum(
                self._var_value(self._mobjs[ni]["var"])*self._last_scalariz["w"][ni]
                    for ni in names )
        elif self._last_scalariz["type"] == "epsilon":
            if not self._q_opt():
                return self._failed_result()
            names = [ self._last_scalariz["main"]
                ] + list(self._last_scalariz["others"].keys())
            scal_val = self._var_value( self._mobjs[names[0]]["var"] )
        elif self._last_scalariz["type"] == "asf":
            if not self._q_opt():
                return self._failed_result()
            names = list(
                set(self._last_scalariz["w"]) | 
                set(self._last_scalariz["obj_bounds"])
                )
            scal_val = self._var_value(self._dummy_asf)
        names = self._parse_names(names) # just sort if names = [0,1,2,...,k-1]
        if x_ is None:
            objs = { ni: self._var_value(self._mobjs[ni]["var"]) for ni in names }
        else:
            objs = { 
                ni: np.einsum("ij,ij->",self._mobjs[ni]["coeffs"], x) 
                    for ni in names }
        obj_out = self._obj2out( self._factor_out(objs) )
        # out objectives -> percentage in [out nadir, out ideal] 
        if self._payoff_status:
            idl, ndr = self.out_ideal, self.out_nadir
            obj_prc = { 
                ni: self._degen_prc if self._mobjs[ni]["degen"] else
                    (obj_out[ni]-ndr[ni]) / (idl[ni] - ndr[ni]) * 100
                        for ni in names 
                    }
            obj_prc_v = obj_dict2vect(obj_prc)
        else:
            obj_prc, obj_prc_v = {}, []
        return {
            "obj": objs,
            "scalar": scal_val,
            "obj v": obj_dict2vect(objs),
            "out": obj_out,
            "out v": obj_dict2vect(obj_out),
            "out %": obj_prc,
            "out % v": obj_prc_v,
            "x": x,
            "status":self._opt_status(),
            "mipgap":self._opt_mipgap(),
            "time":self._opt_time(),          
            }