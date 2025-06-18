import dnn_util as dnn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


#--- Default parameters

# L_error: options for model evaluation: 'L1_abs', 'L1_avg', 'L2'
# descent: options for type of descent:  'L1_abs', 'L1_avg', 'L2'

n = 1000               # number of observations
seed = 565             # for replicability 
eps = 0.000001         # precision on partial derivatives
epochs = 501           # iterations in gradient descent
layers = 3             # number of parameter subsets
learning_rate = 0.1    # learning rate
temperature = 0        #  minimum is 0; large value => more entropy in descent
L_error = 'L2'         # 'L2' minimizes MSE loss, 'L1_abs' minimizes MAE loss
descent = 'L2'         # descent algorithm: 'L1_abs' or 'L2' (best: descent=L_error)
distill_rate = 1.0     # proportions of obs left in data after random deletion
args = {}              # list of hyperparameters passed across all functions
args['eps'] = eps      # for 'L2' descent only
np.random.seed(seed)

# choose type of function
f0 = dnn.f0
f2 = dnn.f2x 
f = f2   

#--- True weights (parameters) & arguments for each function type

if f == f0:    # multivariate curve fitting; dim=nfeatures; multiple layers    

    alpha = 0.10         # may fail with too much noise (alpha = 0.50)
    nfeatures = 7
    centers = 5          # top layers, better if smaller than nfeatures
    layers = 4*centers   # here 4 is the max number of sub-layers per top layer  
    distill_rate = 0.5
    params = np.random.uniform(0, 1, (layers, nfeatures))  
    x = np.random.uniform(0, 1, (nfeatures, n))  
    args['equalize'] = True 
    args['model'] = 'gaussian' # 'approx. gaussian' or 'gaussian' 
   
elif f == f2:    # Nice universal function, 1D for now

    alpha = 0.50    # amount of noise to add to y 
    nfeatures = 70
    distill_rate = 0.50 
    params = np.zeros((layers, nfeatures))
    params[0, :] = np.random.uniform(0.50, 1.00, nfeatures)  
    params[1, :] = np.random.uniform(0.00, 0.95, nfeatures) 
    params[2, :] = np.random.uniform(0.00, 1.00, nfeatures)  
    x = np.random.uniform(0, 100, n)   
    args['equalize'] = True
    args['ghost_params'] = () 

#--- Distillation and adding noise to response y 

y_base = f(params, x, args)                # generate base response
y = np.copy(y_base)
stdev_y = np.std(y)                        # standard deviation of response
y += np.random.normal(0, alpha*stdev_y, n) # add noise to response
(xd, yd) = dnn.distill(distill_rate, x, y) # work on sub-sample of x, y 
np.random.seed(seed)                       # reset seed post-distillation

if f == f0 and nfeatures >= 2:
    plt.tricontour(x[0,:], x[1,:], y, levels=20, cmap='viridis', linewidths=0.5) 
    plt.show()

#--- Main

temperature = 0.30 
(params_estimated, history) = dnn.gradient_descent(epochs, learning_rate,  layers, nfeatures, 
                                               f, yd, xd, temperature, L_error, descent, args)
dnn.summary(x, y, params, params_estimated, f, L_error, history, temperature, args)

#--- Correl y_pred vs y using random weights (base model)

params_test = np.random.uniform(0.5, 1.0, params.shape)
y_pred = f(params_test, x, args)
mae = dnn.loss(f, params_test, y, x, L_error, args)
corr = abs(np.corrcoef(y, y_pred))[0, 1]
print("MAR: %8.5f  Corr: %8.5f" % (mae, corr))

#--- Plot real vs estimated weights

for q in range(layers): 
    label = 'layer ' + str(q)
    plt.scatter(params[q,:], params_estimated[q,:], label = label)
legend = plt.legend()
for text in legend.get_texts():
    text.set_color("white")
plt.show()

