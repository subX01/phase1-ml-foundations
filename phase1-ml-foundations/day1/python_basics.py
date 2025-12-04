# lists can be added in python
a=[1,2,3]
b=[4,5,6]
print(a+b)


import numpy as np

a= np.array([1,2,3])
b= np.array([4,5,6])
print(a+b)

#Understanding shape - Shape tells you how data is arranged 
# Why shape matters- Training of models is not possible if we don't have shapes.
# Matrix multiplication is not possible if we dont have proper shapes 
# Wrong shape- errors during training

# Data	        Shape	Meaning
# [1, 2, 3]	    (3,)	3 values in a line (vector)
# [[1,2],[3,4]]	(2,2)	2 rows, 2 columns (matrix)

import numpy as np;

n= np.array([2,3,5])
print(a)
print("Shape:", a.shape)
