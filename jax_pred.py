import jax.numpy as jnp
from jax import grad

#model prediction function 
def predict(w,x):
    return w*x

x=3.0
y=6.0


#Loss function comparing to true y

def loss_fn(w,x,y):
    y_pred = predict(w,x)
    return (y-y_pred)**2 #squared error 
w=0.5
gradient=grad(loss_fn)(w,x,y) 
print("Gradient:", gradient)

learning_rate= 0.2
new_w= w- learning_rate*gradient
print("updated w:", new_w)
