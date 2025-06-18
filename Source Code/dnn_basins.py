import dnn_util as dnn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# https://github.com/VincentGranville/Machine-Learning/blob/main/Source%20Code/dnn-dev.py #########
# https://github.com/VincentGranville/Machine-Learning/blob/main/Source%20Code/dnn_util.py #####

#--- Parameters

# L_error: options for model evaluation: 'L1_abs', 'L1_avg', 'L2'
# descent: options for type of descent:  'L1_abs', 'L1_avg', 'L2'

n = 150                # number of observations
nfeatures = 5          # 5 or 70 works better than 20?
seed = 565             # for replicability 
eps = 0.000001         # precision on partial derivatives
alpha = 0.10           # amount of noise to add to y  
epochs = 50            # iterations in gradient descent
layers = 3             # number of parameter subsets
learning_rate = 0.1    # learning rate
temperature = 0        #  minimum is 0; large value => more entropy in descent
L_error = 'L2'         # 'L2' minimizes MSE loss, 'L1_abs' minimizes MAE loss
descent = 'L2'         # descent algorithm: 'L1_abs' or 'L2' (best: descent=L_error)
args = {}              # list of hyperparameters passed across all functions
args['eps'] = eps      # for 'L2' descent only
np.random.seed(seed)

f = dnn.f2x    # choose function
params = np.zeros((layers, nfeatures))
params[0, :] = np.random.uniform(0.50, 1.00, nfeatures)  
params[1, :] = np.random.uniform(0.00, 0.95, nfeatures) 
params[2, :] = np.random.uniform(0.00, 1.00, nfeatures)  
x = np.random.uniform(0, 50, n)   
args['equalize'] = True
args['ghost_params'] = ()

#--- Adding noise to response y 

y = f(params, x, args)
y_base = f(params, x, args)                # generate base response
y = np.copy(y_base)
stdev_y = np.std(y)                        # standard deviation of response
y += np.random.normal(0, alpha*stdev_y, n) # add noise to response
np.random.seed(seed)                       # reset seed post-distillation

#--- Main

params_init = np.full((layers, nfeatures), 0.00)
params_init[2, :] = np.full(nfeatures, -2.00)  # way outside range, but works well!
map_p1 = []
map_p2 = []
map_corr = []

for p0 in np.arange(-0.5 ,1, 0.05):
    for p1 in np.arange(-0.5, 1, 0.05):
        params_init[0, :] = np.full(nfeatures, p0) 
        params_init[1, :] = np.full(nfeatures, p1) 
        args['params_init'] = params_init
        (params_estimated, history) = dnn.gradient_descent(epochs, learning_rate,  
                           layers, nfeatures, f, y, x, temperature, L_error, descent, args)
        y_pred = f(params_estimated, x, args)
        corr = abs(np.corrcoef(y, y_pred))[0, 1]
        p_delta = np.mean(params - params_init)
        print("p1: %6.4f p2: %6.4f p_delta: %8.5f corr: %8.5f " % (p0, p1, p_delta, corr))
        map_p1.append(p0)
        map_p2.append(p1)
        map_corr.append(corr*corr)     # this is R squared

#--- Visu

map_corr = np.array(map_corr)
import matplotlib.tri as tri
triang = tri.Triangulation(map_p1, map_p2)
refiner = tri.UniformTriRefiner(triang) 
tri_refi, corr_refi = refiner.refine_field(map_corr, subdiv=3)
corr_refi = (corr_refi - np.min(corr_refi))/(np.max(corr_refi)-np.min(corr_refi))
corr_refi = np.min(map_corr) + (np.max(map_corr)-np.min(corr_refi))*corr_refi
contour = plt.tricontourf(tri_refi, corr_refi, levels = 40, cmap = 'terrain')
plt.colorbar(contour)
plt.show()

#################3 use plots alreadt created // use my interpol method
#########################3 starting with avg true param per layer can work
#########################  50 epochs --> to see what constant p0, p1, p2 quickly lead to good starting point  