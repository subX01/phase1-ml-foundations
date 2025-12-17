import jax.numpy as np

#Datasets
x=np.array([1.,2.,3.,4.,5.]);
y=np.array([2.,4.,6.,8.,10.]);


#Storing the result 
ws=[];
losses=[];

#Using a range of weights this time 
for w in np.arange(-0.1,4.1,0.1):
    y_pred= w * x
    loss=np.mean((y-y_pred)**2)

    ws.append(w)
    losses.append(loss)

#Finding the minimum loss 
min_loss=min(losses)
best_w=ws[losses.index(min_loss)]

print("Minimum loss:", min_loss)
print("Best w is:", best_w);

