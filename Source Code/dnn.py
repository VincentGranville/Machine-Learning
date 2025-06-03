import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.facecolor'] = 'black'

#--- 1. Distillation

def distill(rate, x, y):
    n = len(y)
    nd = int(rate * n)
    chosen = np.random.choice(np.arange(0, n), size=nd, replace=False)
    if len(x.shape) == 2:
        xd = x[:,chosen]  # only keep cols in chosen
    else:
        xd = x[chosen]
    yd = y[chosen]
    print("Distillation factor:", 1-rate) 
    return(xd, yd)


#--- 2. Test functions for curve fitting

def f0(params, x, args=""): 
    model = args['model']
    nfeatures, nobs = x.shape  # nfeatures is m in the paper
    layers, lparams = params.shape 
    ones = np.ones(nobs)
    z = np.zeros(nobs)
    J = layers // 4 
    for j in range(J):
        for k in range(nfeatures):
            theta0 = params[4*j,k]
            theta1 = params[4*j+1,k]*ones
            theta2 = params[4*j+2,k]
            theta3 = 1  # try: 1; params[4*j+3,k]; 1-theta2
            if model == 'approx. gaussian':
                z += theta0*(ones - ((x[k,:]-theta1)*theta2)**2) 
            elif model == 'gaussian':
                z += theta0*np.exp(-(theta3*(x[k,:]-theta1)/theta2)**2) 
    if args['equalize']: 
        z = z - np.min(z)  
    return(z)

def f1(params, x, args = ""): 
    small = args['small'] 
    type = args['function_type']
    xnodes = args['centers']
    model = args['model'] 
    layers, nfeatures = params.shape
    nobs = len(x)
    ones = np.ones(nobs)
    small_ones = small*ones
    z = np.zeros(nobs)
    for k in range(nfeatures):
        if model == 'static':
            # centers are fixed
            xnode_k = xnodes[k]*ones
        elif model == 'latent':
            # centers must be estimated
            xnode_k = params[3,k]*ones
        if type == 'rational':
            z += small*params[0,k]/(params[1,k]*small_ones+((x-xnode_k)/params[2,k])**2) 
        elif type == 'polynom':
            z += params[0,k]*(x-xnode_k)+params[1,k]*(x-xnode_k)**2+params[2,k]*(x-xnode_k)**3
    if args['equalize']:
        z = z - np.min(z)
    return(z)

def f2(params, x, args=""):
    # Dirichet eta function, real part with fixed sigma
    layers, nfeatures = params.shape
    sigma = params[0, 0]
    phi = params[1, 0]
    theta = phi/(1-phi)
    z = 0
    for k in range(nfeatures):
        z += params[2, k] * (-1)**k * np.cos((x+theta)*np.log(k+1)) / (k+1)**(sigma)
    if args['equalize']:
        z = z - np.min(z)
    return(z)

def f3(params, x, args=""):
    # Dirichet eta function, imaginary part with fixed sigma
    layers, nfeatures = params.shape
    sigma = params[0, 0] 
    phi = params[1, 0]
    theta = phi/(1-phi)
    z = 0
    for k in range(nfeatures):
        z += params[2, k] * (-1)**k * np.sin((x+theta)*np.log(k+1)) / (k+1)**(sigma)
    if args['equalize']:
        z = z - np.min(z) 
    return(z)

def g1(rparams, x, args=""):
    small = args['small'] 
    type = args['function_type']
    xnodes = args['centers']
    model = args['model'] 
    layers, nfeatures = rparams.shape
    nobs = len(x)
    z = np.zeros(nobs)
    ones = np.ones(nobs)
    small_ones = small*ones
    for k in range(nfeatures):
        if model == 'static':
            # centers are fixed
            xnode_k = xnodes[k]*ones
        elif model == 'latent':
            # centers must be estimated
            xnode_k = rparams[2,k]*ones
        z += small*rparams[0, k]/(small_ones+((x-xnode_k)/rparams[1, k])**2) 
    if args['equalize']:
        z = z - np.min(z)
    return(z)


#--- 3. Reparameterization

def reparameterize(params):  
    # number of layers in rparams must be <= to that in params
    layers, nfeatures = params.shape
    rparams = np.zeros((layers, nfeatures))
    for k in range(nfeatures): 
        rparams[0, k] = params[0, k]/params[1, k]  
        rparams[1, k] = np.sqrt(params[1, k])*params[2, k]
        rparams[2, k] = params[3, k]  
    return(rparams)  
  
def check_reparametrization(f, g, params, x, args=""):
    y_from_f = f(params, x, args)
    y_from_g = g(reparameterize(params), x, args)
    delta = np.max(abs(y_from_f - y_from_g))
    if delta > 0.000001 or np.isnan(delta):
        print("reparameterization error: delta =", delta)
        exit()
    else:
        print("reparameterization correct: delta =", delta)
    return(delta)


#--- 4. Gradient descent

def loss(f, params_estimated, y, x, mode, args=""):

    y_estimated = f(params_estimated, x, args) 
    z = np.abs(y - y_estimated)
    if mode == 'L1_abs':
        return(np.max(z)) 
    elif mode == 'L1_avg':
        return(np.average(z))
    elif mode == 'L2':
        return(np.dot(z, z)/len(z))

def init_L2_descent(f, y, x, L_error, layers, nfeatures, args):

      params_init = np.full((layers, nfeatures), 0.5)
      init_loss = loss(f, params_init, y, x, L_error, args)
      args['params_init'] = params_init 
      args['init_loss'] = init_loss
      return(init_loss, args)

def init_swarm_descent(f, y, x, L_error, layers, nfeatures, args):

    ntrials = args['ntrials']  # trial called particle in litterature
    loss_swarm = np.zeros(ntrials)
    min_loss = 99999999.99
    params_swarm = []
    for k in range(ntrials):
        params_init = np.zeros((layers, nfeatures))
        params_init[0,:] = np.random.uniform(0.00, 0.50, nfeatures)  # weights
        params_init[1,:] = np.random.uniform(0.25, 1.00, nfeatures)  # offsets
        params_init[2,:] = np.random.uniform(0.25, 1.00, nfeatures)  # skewness
        params_init[3,:] = np.random.uniform(0.00, 1.00, nfeatures)  # centers
        params_swarm.append(params_init)
        loss_val = loss(f, params_init, y, x, L_error, args)
        loss_swarm[k] = loss_val  
        args['params_swarm'] = params_swarm
        args['loss_swarm'] = loss_swarm
        if loss_val < min_loss:
            min_loss = loss_val
            params_best = np.copy(params_init)
    args['params_init'] = np.copy(params_best)
    args['params_estimated'] = np.copy(params_best)
    return(np.min(loss_swarm), args)

def swarm_descent(loss, f, params_estimated, y, x, learning_rate, temp, mode, args=""): 

    # trial (or starting point) is sometimes called particle in swarm optimization
    ntrials = args['ntrials']
    subtrials = args['subtrials']
    params_swarm = args['params_swarm']  # one parameter set per trial
    loss_swarm = args['loss_swarm']      # one loss value per trial
    L_current  = args['current_loss']  
    L_best_global = L_current
    norm = np.sum(abs(f(params_estimated,x, args)))/len(y)
    ones = np.full((layers, nfeatures), 1)

    for trial in range(ntrials):
        L_best = loss_swarm[trial]
        for k in range(subtrials):
            pd = np.random.uniform(-1, 1, (layers, nfeatures)) * norm  
            params_test = params_swarm[trial] - learning_rate * pd
            params_test = np.abs(params_test) 
            L_new = loss(f, params_test, y, x, mode, args)
            if L_new < L_best or np.random.uniform() < temp: 
                L_best = L_new
                params_swarm[trial] = np.copy(params_test)
                loss_swarm[trial] = L_best
        if loss_swarm[trial] < L_best_global: 
            L_best_global = loss_swarm[trial]
            params_estimated = params_swarm[trial]
    return(params_estimated)
    
def partial_derivatives(loss, f, params_estimated, y, x, 
                        learning_rate, temp, L_error, args=""):

    # partial derivatives (pd) of loss function with respect to
    #    the coefficients in params_estimated; use it if mode='L2'

    eps = args['eps']
    layers, nfeatures = params_estimated.shape
    pd = np.zeros((layers, nfeatures))  
    for l in range(layers):
        for k in range(nfeatures):
            params_eps = np.copy(params_estimated) 
            # make eps depend on epoch, layer, parameter ?
            params_eps[l, k] += eps
            L_right = loss(f, params_eps, y, x, L_error, args)
            L_current  = args['current_loss']
            pd[l, k] = (L_right - L_current) / eps
            if np.random.uniform() < temp:  
                pd[l, k] = -pd[l, k]   
    params_new = params_estimated - learning_rate * pd
    for index, value in np.ndenumerate(params_estimated): 
        if 0 < value < 1:  
            # handle situation where params is out of accepted range
            params_estimated[index] = params_new[index]            
    return(params_estimated)

def gradient_descent(epochs, learning_rate, layers, nfeatures, f, y, x, 
                     temperature, L_error, descent, args=""):
    n = len(y)
    epoch = 0
    history = []
    if descent == 'L2':
        (mae, args) = init_L2_descent(f, y, x, L_error, layers, nfeatures, args)
    elif descent == 'swarm_descent':
        (mae, args) = init_swarm_descent(f, y, x, L_error, layers, nfeatures, args)
    history.append(mae)
    args['current_loss'] = history[-1]
    params_estimated = args['params_init']

    while epoch < epochs:
        decay = history[-1] 
        temp = temperature * decay         
        if descent == 'swarm_descent':
            params_estimated = swarm_descent(loss, f, params_estimated, y, x,
                                       learning_rate, temp, L_error, args)
        elif descent == 'L2':
            params_estimated = partial_derivatives(loss, f, params_estimated, y, x, 
                                       learning_rate, temp, L_error, args)
        mae = loss(f, params_estimated, y, x, L_error, args) 
        history.append(mae)   
        args['current_loss'] = history[-1]
        if epoch % 100 == 0:
            print("Epoch %5d MAE %8.5f" %(epoch, mae))
        epoch += 1
    return(params_estimated, history)


#--- 5. Visualization

def summary(x, y, params, params_estimated, f, L_error, args):

    y_estimated = f(params_estimated, x, args)
    visu(history, y, x, f, params_estimated, params, args)
    mae_full_data = loss(f, params_estimated, y, x, L_error, args)

    corr = abs(np.corrcoef(y, y_estimated))[0, 1]
    print("mae_d: %8.5f mae_f: %8.5f corr: %8.5f temp: %6.4f function: %s (n=%5d)"
             % (history[-1], mae_full_data, corr, temperature, 'f', len(y)))
    return()

def visu(history, y, x, f, params_estimated, params, args):  

    xvalues = np.arange(len(history))
    plt.plot(xvalues, history, linewidth = 0.6, c='gold')
    plt.show()
    y_estimated = f(params_estimated, x, args)    
    plt.scatter(y, y_estimated, s = 0.8, c='gray')
    ymin = np.min(y)
    ymax = np.max(y) 
    plt.plot((ymin, ymax), (ymin, ymax), c='red', linewidth = 0.8, alpha=1.0)
    plt.show()
    if len(x.shape) == 1:
        # input data is 1-Dim
        xvalues = np.arange(np.min(x), np.max(x), 0.001)
        y_base = f(params, xvalues, args)
        y_reconstructed = f(params_estimated, xvalues, args)
        plt.plot(xvalues, y_base, c='red', linewidth = 0.8, alpha = 0.8)
        plt.plot(xvalues, y_reconstructed, c='green', linewidth = 0.8, alpha = 0.8)
        plt.scatter(x, y, c='gold', s=1.5)
        plt.show()    
    return()

def visu_eta_orbit(f2, f3, x, y, params, params_estimated, args):

    xmin, xmax = np.min(x), np.max(x)
    step = (xmax - xmin)/ 5000 
    xvalues = np.arange(xmin, xmax, step)
    y_exact_real = f2(params, xvalues, args) 
    y_reconstructed_real = f2(params_estimated, xvalues, args)
    y_exact_imaginary = f3(params, xvalues, args) 
    y_reconstructed_imaginary = f3(params_estimated, xvalues, args)
    y_estimated_imaginary = f3(params_estimated, x, args)
    plt.scatter(y_exact_real, y_exact_imaginary, s=0.08, c='red')
    plt.scatter(y_reconstructed_real, y_reconstructed_imaginary, s=0.08, c='green')
    for k in range(len(xvalues)):
        if k %5 == 0:
            exact = (y_exact_real[k], y_exact_imaginary[k])
            reconstructed = (y_reconstructed_real[k], y_reconstructed_imaginary[k])
            plt.plot((exact[0], reconstructed[0]), (exact[1],reconstructed[1]), c='gold', linewidth = 0.4)
    plt.show()
    return()


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
f = f2    # choices: f0, f1, f2

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
    args['ntrials'] = 4 ## 2  #10 # for descent='swarm_descent'; large ==> fewer epochs
    args['subtrials'] = 50 ## 100 #10 # number of particles in 'swarm_descent' descent

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
(xd, yd) = distill(distill_rate, x, y)     # work on sub-sample of x, y 
np.random.seed(seed)                       # reset seed post-distillation

if f == f0 and nfeatures >= 2:
    plt.tricontour(x[0,:], x[1,:], y, levels=20, cmap='viridis', linewidths=0.5) ##100 levels
    plt.show()
elif f == f1:
    plt.plot(x, y, c='gold', linewidth = 0.8, alpha = 0.8)
    plt.axhline(y = 0, color='gray', linewidth = 0.8, linestyle='--')
    plt.show()


#--- 7. Main: without reparameterization

temperature = 0.00
(params_estimated, history) = gradient_descent(epochs, learning_rate,  layers, nfeatures, 
                                               f, yd, xd, temperature, L_error, descent, args)
summary(x, y, params, params_estimated, f, L_error, args)
if f == f2:
    visu_eta_orbit(f2, f3, x, y, params, params_estimated, args)

temperature = 0.30
(params_estimated, history) = gradient_descent(epochs, learning_rate,  layers, nfeatures, 
                                               f, yd, xd, temperature, L_error, descent, args)
summary(x, y, params, params_estimated, f, L_error, args)
if f == f2:
    visu_eta_orbit(f2, f3, x, y, params, params_estimated, args)

#--- 8. Main: with reparameterization

if f == f1:

    # g1 = f1 but reparameterized, using 3 layers rather than 4
    check_reparametrization(f1, g1, params, x, args)   # exit if incorrect reparameterization
    rparams = reparameterize(params)

    temperature = 0.00
    (rparams_estimated, history) = gradient_descent(epochs, learning_rate, layers, nfeatures, 
                                                    g1, yd, xd, temperature, L_error, descent, args) 
    summary(x, y, rparams, rparams_estimated, g1, L_error, args)

