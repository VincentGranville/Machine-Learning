import dnn_util as dnn
import numpy as np

# first param in each layer is ghost parameter (ignored) 

params = [
[ 0.85165599, 0.65393516, 0.90083926, 0.57970312, 0.5946254 , 0.73643363,
  0.55756234, 0.64186534, 0.80801472, 0.82606374, 0.71499897, 0.73555014,
  0.62021517, 0.6268527 , 0.65796546, 0.59863172, 0.71039765, 0.90789613,
  0.94014816, 0.72732649, 0.78167293, 0.92641785, 0.93591317, 0.96945629,
  0.68613199, 0.93763731, 0.86195935, 0.81227456, 0.79719039, 0.81208008,
  0.72614698, 0.92174495, 0.58862909, 0.62406524, 0.86434046, 0.86703429,
  0.94441231, 0.88115144, 0.81335939, 0.65105365, 0.94022282, 0.84201915,
  0.71538169, 0.83528606, 0.6441331 , 0.7852612 , 0.66220926, 0.82515595,
  0.65795606, 0.67691099, 0.77812643, 0.98273006, 0.59207137, 0.68811504,
  0.75861639, 0.73452121, 0.74517432, 0.65203426, 0.75770495, 0.7816799 ,
  0.57202272, 0.80374978, 0.71046679, 0.83372658, 0.50691402, 0.78275801,
  0.60133686, 0.89723749, 0.77287982, 0.64966611],
[ 0.03272436, 0.60068094, 0.52481526, 0.10941128, 0.28148083, 0.05686739,
  0.73901658, 0.5711666 , 0.43208521, 0.54247193, 0.24337865, 0.40012704,
  0.66666713, 0.67280809, 0.53716467, 0.92670779, 0.70601287, 0.38841128,
  0.55979079, 0.02678779, 0.9061923 , 0.05652429, 0.76988525, 0.7568919 ,
  0.56452636, 0.35434107, 0.12658475, 0.60042097, 0.29196158, 0.36077978,
  0.13398964, 0.43216379, 0.10449703, 0.90678883, 0.88102394, 0.71936334,
  0.42029974, 0.71675544, 0.00712738, 0.79391547, 0.81995011, 0.05606797,
  0.25297917, 0.72612892, 0.21482573, 0.24955405, 0.31215618, 0.24015214,
  0.45946812, 0.43925987, 0.46615066, 0.69840377, 0.48261149, 0.27981838,
  0.88301791, 0.6824583 , 0.51275078, 0.69004386, 0.89255255, 0.28026081,
  0.78764299, 0.74846045, 0.89357187, 0.1035059 , 0.43987066, 0.16285935,
  0.15217154, 0.14389275, 0.3582901 , 0.15766135,],
[ 0.01419936, 0.66009896, 0.38206559, 0.78682221, 0.03444658, 0.99731197,
  0.51496302, 0.78708706, 0.80123275, 0.89435317, 0.34394872, 0.46695414,
  0.19305834, 0.47183516, 0.13496453, 0.18653406, 0.0712197 , 0.21639504,
  0.47479447, 0.09374338, 0.05715193, 0.77836343, 0.0124174 , 0.62569663,
  0.85591189, 0.9926094 , 0.07756657, 0.12693357, 0.39481386, 0.38222598,
  0.92881001, 0.51188613, 0.22093432, 0.04217101, 0.02322436, 0.84208532,
  0.917206  , 0.87181315, 0.37480809, 0.72984915, 0.70780806, 0.17890348,
  0.15034743, 0.82988439, 0.45186137, 0.54579097, 0.13392343, 0.78375693,
  0.93555226, 0.40433637, 0.84696389, 0.42866298, 0.2198815 , 0.25205698,
  0.77889308, 0.27655477, 0.85801324, 0.63197665, 0.84314224, 0.4005952 ,
  0.53954967, 0.00716186, 0.30784849, 0.46806576, 0.61351027, 0.68673918,
  0.12131183, 0.37190554, 0.23941732, 0.71112032]
]

params_estimated = [
 [ 5.00000000e-01,  6.57205723e-01,  5.77373222e-01,  4.26696092e-01, 
   7.35959399e-01,  5.21481186e-01,  5.30906622e-01,  4.71189238e-01, 
   6.40611112e-01,  6.11135813e-01,  5.10081658e-01,  7.65538097e-01, 
   6.83448016e-01,  5.83517032e-01,  8.14389630e-01,  6.43282517e-01, 
   8.07467236e-01,  6.80919030e-01,  6.57909168e-01,  7.25275529e-01, 
   7.82234131e-01,  7.21306039e-01,  6.76627816e-01,  5.78989792e-01, 
   8.34201234e-01,  7.04253624e-01,  7.28066131e-01,  7.73784287e-01, 
   7.19572549e-01,  6.85601446e-01,  4.14769384e-01,  7.96142015e-01, 
   5.93859380e-01,  7.36236845e-01,  7.53390172e-01,  6.79515577e-01, 
   6.62614110e-01,  6.92962450e-01,  6.67784466e-01,  4.58681521e-01, 
   4.88448016e-01,  4.88723191e-01,  5.02802077e-01,  4.83484009e-01, 
   6.14140150e-01,  5.38363466e-01,  6.07040618e-01,  6.13644218e-01, 
   5.69215115e-01,  5.57758795e-01,  5.45466745e-01,  5.78508679e-01, 
   5.72180951e-01,  5.53781884e-01,  6.19279960e-01,  5.34362066e-01, 
   5.26734488e-01,  5.72738056e-01,  4.75992406e-01,  6.07747426e-01, 
   5.78616656e-01,  5.22134307e-01,  6.10554347e-01,  3.77547161e-01, 
   6.54105706e-01,  7.05287295e-01,  4.75996247e-01,  6.54015623e-01, 
   5.66095436e-01,  5.05745176e-01],
 [ 5.00000000e-01,  5.83833893e-01,  4.91904403e-01,  1.25438603e-01,
   6.35464197e-01,  1.10297326e-01, -3.75237974e-03,  5.69711860e-01,
   4.05103681e-01,  5.51372673e-01,  2.32098212e-01,  3.94933327e-01,
  -9.70877197e-04, -1.05474933e-03, -2.64338049e+00,  5.59912556e-01,
   1.56606050e-01,  4.05079564e-01,  5.40236360e-01, -3.67635996e-05,
   1.36488794e+00,  6.47815525e-01,  2.37767315e-01,  5.20612941e-01,
   2.28740696e+00,  5.93431687e-01,  4.89158909e-01,  2.63786590e-01,
   2.40962012e-01,  4.84697015e-01,  2.08340344e-01, -1.37374885e-01,
   1.74815761e-01,  1.36428353e+00,  4.13964617e-01,  4.52614297e-01,
   4.32182471e-01,  3.89270985e-01,  3.71256166e-01,  2.43289148e-01,
   4.55327234e-01,  1.88734976e-01,  4.09421020e-01,  3.82752743e-01,
   2.75456633e-01,  2.63888594e-01,  5.46653108e-01,  3.78859553e-01,
   4.94729241e-01,  3.94001261e-01,  2.64271968e-01,  2.90665456e-01,
   3.50218501e-01,  3.11148402e-01,  4.19101037e-01,  3.11960944e-01,
   5.20149099e-01, -1.31538616e+00,  4.62896338e-01,  3.69567055e-01,
   4.30300269e-01,  3.63124105e-01,  4.01037372e-01,  3.02719796e-01,
   3.42525828e-01,  1.13817984e+00,  3.42258696e-01,  5.06025185e-01,
   4.26860279e-01,  1.27410709e-01],
 [ 5.00000000e-01,  6.45864477e-01,  2.33655768e-01,  6.21635598e-01,
   8.99780033e-02,  6.14990066e-01,  4.40293493e-01,  5.69019403e-01,
   4.75859584e-01,  5.33847787e-01,  3.31334716e-01,  4.56083296e-01,
   2.06450798e-01,  3.32774263e-01,  8.86047271e-02,  3.18966506e-01,
   3.10782892e-01,  3.36909104e-01,  3.86074774e-01,  1.30867215e-01,
   1.54890207e-01,  2.62155653e-01,  2.89148264e-01,  4.02052731e-01,
   2.80438164e-01,  3.27518555e-01,  4.53911786e-01,  3.24543634e-01,
   3.65356258e-01,  3.78658632e-01,  5.48027306e-01,  3.91882204e-01,
   4.03089716e-01,  3.32519337e-01,  3.66017910e-01,  3.86914047e-01,
   4.22057811e-01,  3.72911299e-01,  3.60804467e-01,  4.94378382e-01,
   4.98626530e-01,  4.31883071e-01,  4.96857436e-01,  4.06386395e-01,
   4.69651246e-01,  5.04750676e-01,  5.15184776e-01,  4.27383923e-01,
   5.08758114e-01,  4.42754161e-01,  4.97179009e-01,  4.50025884e-01,
   4.53306723e-01,  4.21162807e-01,  3.91604992e-01,  4.77772938e-01,
   4.60409364e-01,  4.90837158e-01,  5.04154223e-01,  4.79523086e-01,
   5.10390306e-01,  4.51741320e-01,  4.26366445e-01,  5.32334118e-01,
   4.28430678e-01,  4.30619688e-01,  5.28295919e-01,  4.68558315e-01,
   4.86573051e-01,  5.07028625e-01]
]

n = 500
f = dnn.f2x
seed = 4769
np.random.seed(seed)
args = {}
args['equalize'] = True 
args['ghost_params'] = () 
params = np.array(params)
params_estimated = np.array(params_estimated)
layers, nfeatures = params_estimated.shape

x = np.random.uniform(0, 100, n)   
y = f(params, x, args)
y_pred = f(params_estimated, x, args)

import matplotlib.pyplot as plt
import matplotlib as mpl

#--- 1. Show model fit

L_error = 'L2'
history = [0,]
temperature = 0
dnn.summary(x, y, params, params_estimated, f, L_error, history, temperature, args)
      
#--- 2. Weight distillation (ghosting)

wlist = range(nfeatures)
xghost = []
yghost = []
hash_min = {}
hash_max = {}

for j in range(1, nfeatures-1):

   ntests = 2000  # to debug, use ntests = 20
   min_corr = 1
   max_corr = -1
   hash_min[j] = ""
   hash_max[j] = ""
   
   for test in range(ntests): 
       # ghost j*layers parameters, out of nfeatures*layers
       ghosted = np.random.choice(wlist, size=j, replace=False)
       params_test = np.copy(params_estimated)
       for k in ghosted:
           for l in range(layers):
               params_test[l,k] = 0
       y_pred = f(params_test, x, args)
       corr = abs(np.corrcoef(y, y_pred))[0, 1]
       xghost.append(j)
       yghost.append(corr*corr)  # R-squared
       if corr < min_corr:
           min_corr = corr
           hash_min[j] = (corr, ghosted)
       if corr > max_corr:
           max_corr = corr
           hash_max[j] = (corr, ghosted)
       
plt.scatter(xghost, yghost, s=0.1, c='gold')
plt.show()

for j in range(1, nfeatures-1):
    min_vector  = hash_min[j]
    corr_min    = min_vector[0]
    ghosted_min = min_vector[1] 
    max_vector  = hash_max[j]
    corr_max    = max_vector[0]
    ghosted_max = max_vector[1]
    print("\nNumber of ghosted features", j)
    print("Min corr:", corr_min, "\n", "Ghosted:\n", ghosted_min)
    print("Max corr:", corr_max, "\n", "Ghosted:\n", ghosted_max)

#--- 3. Weight blurring

xblurr = []
yblurr = []
xmeanblurr = []
ymeanblurr = []
for blurr_rate in np.arange(0, 1.0, 0.002):
    mean_R2 = 0
    ntests = 500  # to debug, use ntests = 5
    for test in range(ntests):
        blurr_matrix = np.random.uniform(-blurr_rate, blurr_rate, (layers, nfeatures))
        params_test = params_estimated + blurr_matrix
        y_pred = f(params_test, x, args)
        corr = abs(np.corrcoef(y, y_pred))[0, 1]
        xblurr.append(blurr_rate)
        yblurr.append(corr*corr)  # R-squared
        mean_R2+= corr*corr
    xmeanblurr.append(blurr_rate)
    ymeanblurr.append(mean_R2/ntests)

plt.scatter(xblurr, yblurr, s=0.1, c='gold')
plt.plot(xmeanblurr, ymeanblurr, c='red', linewidth=0.8)
plt.show()

#--- 4. Model watermarking

def distance_to_codes(params, wm_codes):
    nearest_neighbor = {}
    nearest_neighbor_dist = {}
    layers, nfeatures = params.shape
    ncodes = len(wm_codes)
    for k in range(nfeatures):
        for l in range(layers):
            min_dist = 9999
            for j in range(ncodes):
                dist = abs(params[l, k] - wm_codes[j])
                if dist < min_dist:
                    min_dist = dist
                    nearest_neighbor[(l, k)] = j
                    nearest_neighbor_dist[(l, k)] = min_dist
    sorted_items = sorted(nearest_neighbor_dist.items(), key=lambda item: item[1])
    nearest_neighbor_dist = dict(sorted_items)
    return(nearest_neighbor, nearest_neighbor_dist)

def smallest_dist_lookup(nearest_neighbor, nearest_neighbor_dist, wm_codes, params, n_wm):
    count = 0
    print("\nparam idx | code idx | dist | code | param")
    for key in nearest_neighbor_dist:
        dist = nearest_neighbor_dist[key]
        wm_code_idx = nearest_neighbor[key]
        wm_code = wm_codes[wm_code_idx]
        if count < n_wm + 5:
            layer = key[0]
            feature = key[1]
            param = params[layer, feature]
            print("%2d %4d %4d %10.9f %10.9f %10.9f" %(layer, feature, wm_code_idx, dist, wm_code, param)) 
            count += 1
    return()

#- 4.1. Create params_test as params_estimated with n_wm params substituted by closes watermark codes

ncodes = 500  # total number of available watermark codes
n_wm = 10     # number of parameters replaced by closest watermark code
wm_codes = np.random.uniform(0, 1, ncodes)
nearest_neighbor, nearest_neighbor_dist = distance_to_codes(params_estimated, wm_codes)
smallest_dist_lookup(nearest_neighbor, nearest_neighbor_dist, wm_codes, params_estimated, n_wm)

params_test = np.copy(params_estimated)
count = 0
used_code = {}

for key in nearest_neighbor_dist:
    dist = nearest_neighbor_dist[key]
    wm_code_idx = nearest_neighbor[key]
    wm_code = wm_codes[wm_code_idx]
    if count < n_wm and wm_code not in used_code:
        layer = key[0]
        feature = key[1]
        params_test[layer, feature] = wm_code
        used_code[wm_code] = True
        count += 1

#- 4.2. Test params_test for presence of watermarks

nn_list, nn_dist = distance_to_codes(params_test, wm_codes)
smallest_dist_lookup(nn_list, nn_dist, wm_codes, params_test, n_wm)

y_pred = f(params_estimated, x, args)
y_pred_wm = f(params_test, x, args)
corr_wm = abs(np.corrcoef(y_pred, y_pred_wm))[0, 1]
print("\nCorr b/w y_pred and y_pred_wm", corr_wm) 
