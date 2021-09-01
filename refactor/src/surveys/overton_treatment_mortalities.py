"""
This is a self-contained script with the sole scope of providing a simple background for
the fish background mortality formulae from the Overton et al. (2018) paper.
For more information, see  https://doi.org/10.1111/raq.12299

We assume that no treatment has _per se_ any ability to kill a fish by intoxication as long
as dosages are respected. At most, treatment-induced mortality is caused by increased stress
levels, loss of appetite, reduced breathing capacity and so on.
Therefore, the values shown in the paper use the pp unit with respect to the overall
fish mortality percentage increase from the earlier month where no treatment was applied.

Data is extracted from Figure 6 (i) and processed through a L2-regularised linear regressor.
For simplicity, we model this as a quadratic

To run this file one needs sklearn
"""


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

pp_bins = np.array([1, 2.5, 5, 10, 25])
temperatures = np.array([0, 4, 7, 10, 13])

# First axis: mass (<= 2Kg, > 2Kg)
# Second axis: temperature bin
# Third axis: pp increase bin
data = np.array([[[3, 3.5, 4, 0, 0],
                  [4.5, 0, 0.5, 0, 0],
                  [6, 5, 0.5, 0, 0],
                  [5.5, 4, 2, 0.5, 0.5],
                  [5, 3, 1, 0, 0]],
                 [[5, 3, 0, 0, 0],
                  [3, 1, 0.5, 0, 0],
                  [4.5, 2, 2, 1.5, 0],
                  [5, 2.5, 2, 2, 0],
                  [7.5, 4.5, 1, 0.5, 0]]])

normalised_pp_increase = (np.sum(data * pp_bins, axis=2) / np.sum(data, axis=2)).flatten()
mass_indicator = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X = np.c_[np.tile(temperatures, 2), mass_indicator]

pipeline = Pipeline([("poly", PolynomialFeatures(degree=2)),
                     ("ridge", Ridge())])

trained_model = pipeline.fit(X, normalised_pp_increase)
print(trained_model.steps[1][1].coef_)