import jax.numpy as np;from jax import grad
x=np.array([1,2,3,4])
y=np.array([2,4,6,8])
w=0.2

def pred(w,x):
    return x*w

learning_rate=0.01
def loss_fn(w,x,y):
    y_pred=pred(w,x)
    return np.mean((y-y_pred)**2)

for i in range(10):
    gradient= grad(loss_fn)(w,x,y)
    w=w-learning_rate*gradient
    print("updated w:",w)

