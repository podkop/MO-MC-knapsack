# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:29:08 2020

@author: podkop
"""

import pandas as pd
import numpy as np

ex_df = pd.read_excel("frs_random.xlsx",header=[0,1])
crit_names = ex_df.columns.get_level_values(0).unique()[1:]

## Data structure

with open("frs_input.py","w") as f:
    np.set_printoptions(suppress=True,linewidth = 10**10)
    f.write(
        "crit_names = " + np.array2string(
            crit_names.to_numpy(),
            separator = ","
            ) + "\n"
        )
    f.write(
        "regime_names = " +
        np.array2string(
            ex_df[crit_names[0]].columns.to_numpy(),
            separator = ","
            ) +
        "\n"
        )
    f.write(
        "areas = " +
        np.array2string(
            ex_df.iloc[:,0].to_numpy(),
            separator = ","
            ) +
        "\n"
        )
    f.write(
        "coeffs = "+
            np.array2string(
                -np.array([ ex_df[ci].values for ci in crit_names ]),
            separator = ","
            ).replace("\n","")
        )