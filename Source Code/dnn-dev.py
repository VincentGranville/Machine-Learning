import dnn_util as dnn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#--- 6. Initializations

#-- 6.1. default parameters

# L_error: options for model evaluation: 'L1_abs', 'L1_avg', 'L2'
# descent: options for type of descent:  'L1_abs', 'L1_avg', 'L2'

n = 300                # number of observations
nfeatures = 10         # number of features
seed = 565             # for replicability 
eps = 0.000001         # precision on partial derivatives
alpha = 0.10           # amount of noise to add to y
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
f1 = dnn.f1
f2 = dnn.f2 
f3 = dnn.f3
f = f1   

#-- 6.2. parameters & arguments for each function type

if f == f0:    # multivariate curve fitting; dim=nfeatures; multiple layers    

    nfeatures = 7
    centers = 5          # top layers, better if smaller than nfeatures
    layers = 4*centers   # here 4 is the max number of sub-layers per top layer  
    distill_rate = 0.5
    params = np.random.uniform(0, 1, (layers, nfeatures))  
    x = np.random.uniform(0, 1, (nfeatures, n))  
    args['equalize'] = True 
    args['model'] = 'gaussian' # 'approx. gaussian' or 'gaussian' 
   
elif f == f1:    # curve fitting in 1D, multiple layers; chaotic + singularities

    nfeatures = 12   # the number of centers in the parameter space
    L_error = 'L2'     
    descent = 'swarm_descent'
    layers = 4 
    params = np.zeros((layers, nfeatures))    
    params[0,:] = np.random.uniform(0.25, 0.50, nfeatures)     # weights
    params[1,:] = np.random.uniform(0.25, 1.00, nfeatures)     # offset
    params[2,:] = np.random.uniform(0.25, 1.00, nfeatures)     # skewness
    params[3,:] = np.random.uniform(0.00, 1.00, nfeatures)     # centers
    params[0,:4] = [0, 0, 0 , 0]       # first 4 features not used for f, only for DNN 
    x = np.linspace(0, 1, num=n)       # generate observations
    args['equalize'] = False
    args['small']    = 0.001           # > 0; singularity if small=0
    args['function_type'] = 'rational' # 'rational' or 'polynom'  
    args['centers'] = params[3, :]     # used as fixed centers in static model
    args['model']   = 'latent'         # 'latent' (moving centers) or 'static'
    args['ntrials'] = 4                # for descent='swarm_descent'; large ==> fewer epochs
    args['subtrials'] = 50             # number of particles in 'swarm_descent' descent

elif f == f2:    # Using Riemann zeta function to model and predict orbits

    nfeatures = 30
    params = np.zeros((layers, nfeatures))
    params[0, 0] = 0.75  
    params[1, 0] = 0.10 
    for k in range(nfeatures):
        params[2, k] = 1
    x = np.random.uniform(0, 25, n)   
    args['equalize'] = True

#-- 6.3. distillation, and adding noise to response y 

y_base = f(params, x, args)                # generate base response
y = np.copy(y_base)
stdev_y = np.std(y)                        # standard deviation of response
y += np.random.normal(0, alpha*stdev_y, n) # add noise to response
(xd, yd) = dnn.distill(distill_rate, x, y) # work on sub-sample of x, y 
np.random.seed(seed)                       # reset seed post-distillation

if f == f0 and nfeatures >= 2:
    plt.tricontour(x[0,:], x[1,:], y, levels=20, cmap='viridis', linewidths=0.5) 
    plt.show()
elif f == f1:
    plt.plot(x, y, c='gold', linewidth = 0.8, alpha = 0.8)
    plt.axhline(y = 0, color='gray', linewidth = 0.8, linestyle='--')
    plt.show()

#--- 7. Main: without reparameterization

temperature = 0.00
(params_estimated, history) = dnn.gradient_descent(epochs, learning_rate,  layers, nfeatures, 
                                               f, yd, xd, temperature, L_error, descent, args)
dnn.summary(x, y, params, params_estimated, f, L_error, history, temperature, args)
if f == f2:
    dnn.visu_eta_orbit(f2, f3, x, y, params, params_estimated, args)

temperature = 0.30
(params_estimated, history) = dnn.gradient_descent(epochs, learning_rate,  layers, nfeatures, 
                                               f, yd, xd, temperature, L_error, descent, args)
dnn.summary(x, y, params, params_estimated, f, L_error, history, temperature, args)
if f == f2:
    dnn.visu_eta_orbit(f2, f3, x, y, params, params_estimated, args)

#--- 8. Main: with reparameterization

if f == f1:

    # g1 = f1 but reparameterized, using 3 layers rather than 4
    g1 = dnn.g1
    dnn.check_reparametrization(f1, g1, params, x, args)   # exit if incorrect reparameterization
    rparams = dnn.reparameterize(params)

    temperature = 0.00
    (rparams_estimated, history) = dnn.gradient_descent(epochs, learning_rate, layers, nfeatures, 
                                                    g1, yd, xd, temperature, L_error, descent, args) 
    dnn.summary(x, y, rparams, rparams_estimated, g1, L_error, history, temperature, args)

