# Linear Regression=> Linear regression is making the model train on the formula of y=w*x

# Here, x is the input variable, y is the output variable, w is the weight of the variable(what model learns)
# if the error is positive, we should go up 
# if the error is negative, we should go down

# if the predictions are too low => we should go up
# if the predictions are too high => we should go down

#epoch=learning through the dataset one at a time
# predict → calculate error → update w → repeat




import numpy as np

x= np.array([2,3,5], dtype=float)
y= np.array([3,5,7], dtype=float)

w=0.0 #initial weight
for epoch in range(100):
    y_pred= w*x
    error= y- y_pred
    w= w+0.1 * (error * x).mean()
    print(epoch, w)
print("Final weight:",w)        
print("Predictions:", w*x)





import numpy as np;

a=np.array([1,3,4,5], datatype= float)
b=np.array([2,4,5,6], datatype=float)
w=0.0 #initial data 

for epoch in range(20):

    y_pred= x* w
    error= y- y_pred
    w= w+ 0.1*(error*x).mean()
    print(epoch, w)
    