import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip  # to produce mp4 video
from PIL import Image  # for some basic image processing

def fit_ellipse(x, y):

    # Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    # the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    # arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    # Based on the algorithm of Halir and Flusser, "Numerically stable direct
    # least squares fitting of ellipses'.

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def cart_to_pol(coeffs):

    # Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    # ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    # The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    # ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    # respectively; e is the eccentricity; and phi is the rotation of the semi-
    # major axis from the x-axis.

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def sample_from_ellipse(x0, y0, ap, bp, phi):

    x=np.empty(npts)
    y=np.empty(npts)

    # sample from multivariate normal, then rescale 
    cov=[[ap,0],[0,bp]]
    u, v = np.random.multivariate_normal([0, 0], cov, size = npts).T
    d=np.sqrt(u*u/(ap*ap) + v*v/(bp*bp))
    u=u/d
    v=v/d
    angle=np.arctan2(u,v)
    x_unsorted=x0+np.cos(phi)*u-np.sin(phi)*v
    y_unsorted=y0+np.sin(phi)*u+np.cos(phi)*v

    # sort the points x, y for nice rendering with mpl.plot
    hash={}
    hash = dict(enumerate(angle.flatten(), 0)) # convert angle to dictionary
    idx=0
    for w in sorted(hash, key=hash.get):
        x[idx]=x_unsorted[w]
        y[idx]=y_unsorted[w]
        idx=idx+1

    return x, y

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi, sampling='Standard'):

    # Return npts points on the ellipse described by the params = x0, y0, ap,
    # bp, e, phi for values of the parametric variable t between tmin and tmax.

    x0, y0, ap, bp, e, phi = params
    
################## need better sampling
### t random on 0,2 pi // distance to center must be uniformly distributed ???
### pick t evenly on circle the map to ellipse??? 
### this oversample far from center, undersample close to center
#### use rejection sampling: sample dist to center uniformly on 0, maxDist, get (x, y) with that distance

### sampling on an ellipse: https://math.stackexchange.com/questions/2059764/uniform-sampling-of-points-on-the-curve-of-an-ellipse-jacobian-of-transformatio
### https://math.stackexchange.com/questions/973101/how-to-generate-points-uniformly-distributed-on-the-surface-of-an-ellipsoid

    if sampling=='Standard':
        t = np.linspace(tmin, tmax, npts)
        x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)

    elif sampling=='Enhanced':
        x, y = sample_from_ellipse(x0, y0, ap, bp, phi)

        # The npts points (x, y) are obtained by rejection sampling, to make sure 
        # the distances to x0, y0 are uniformy distributed  
        #count=0
        ### trial=0
        #x=np.empty(npts)
        #y=np.empty(npts)
        #z={}
        #distMax=max(ap,bp)**2  ############## 30+max(ap,bp)**2
        #while count < npts:
           ### trial=trial+1
         #  t=np.random.uniform(tmin,tmax)
         #  u = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
         #  v = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
           ### dist = np.sqrt((u-x0)*(u-x0)+(v-y0)*(v-y0)) ###################
          # dist = (u-x0)*(u-x0)+(v-y0)*(v-y0) #########################3
          # distRand=np.random.uniform(0,distMax)
          # if dist < distRand:  # accept sampled point ############### include points with distMax to avoid gaps when plotting
          #     z[t]=[(u,v)]
          #     count=count+1
          #     if frame==nframes-1:  ####
          #         print(t,dist,distMax)  ####
        # sort x, y according to z keys so points can be connected by lines in plt.plot
        #count=0
        #for t in sorted(z.keys()):
         #   [(x[count], y[count])] = z[t]
         #   count=count+1

    if frame==nframes-1:  #####
      print("***** A sampling=",sampling) ####
      out=open("test.txt","w") ######## ------------- issue when frame==nframes-1 
      for kk in range(0,npts): ############
        line=str(x[kk])+"\t"+str(y[kk])+"\n" ####
        out.write(line) ####
      out.close() ####
    #exit()

    return x, y

def main(npts, noise, seed, tmin, tmax, params, sampling):

    x0, y0, ap, bp, phi = params  
    # Get some points x, y on the ellipse (no need to specify the eccentricity).
    x, y = get_ellipse_pts((x0, y0, ap, bp, None, phi), npts, tmin, tmax)
    
    # perturb x, y on the ellipse with some noise
    if frame==nframes-1:      
        noise=0   # to produce the exact curve in last frame 
    np.random.seed(seed)
    if noise_CDF=='Normal':
      cov = [[1,0],[0,1]]  
      u, v = np.random.multivariate_normal([0, 0], cov, size = npts).T
      x += noise * u
      y += noise * v
    elif noise_CDF=='Uniform':
      x += noise * np.random.uniform(-1,1,size=npts) 
      y += noise * np.random.uniform(-1,1,size=npts)

    coeffs = fit_ellipse(x, y)

    # print exact and estimated values or curve parameters
    phi2 = phi % np.pi  # make sure angle phi is between 0 and pi
    print('Exact  x0, y0, ap, bp, phi = : %+.5f %+.5f %+.5f %+.5f %+.5f' % (x0,y0,ap,bp,phi2))
    x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
    phi2 = phi % np.pi    # make sure angle phi is between 0 and pi
    print('Fitted x0, y0, ap, bp, phi = : %+.5f %+.5f %+.5f %+.5f %+.5f' % (x0,y0,ap,bp,phi2))

    # intialize plotting parameters
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rc('axes',edgecolor='black') # border color
    plt.rc('xtick', labelsize=6) # font size, x axis  
    plt.rc('ytick', labelsize=6) # font size, y axis
    if frame==nframes-1:
        col='black' # color of exact curve
        alpha=1    # color transparency level, exact curve
    else:
        col='blue'  # color of fitted curve
        alpha=0.05  # transparency level

    # produce plot, save it as an image with filename image 
    plt.scatter(x, y,s=0.5,color='red',alpha=0.03)   # plot training set points in red
    x, y = get_ellipse_pts((x0, y0, ap, bp, e, phi),npts, tmin, tmax, sampling)
    plt.plot(x, y, linewidth=0.5, color=col,alpha=alpha) # plot fitting curve 
    plt.savefig(image, bbox_inches='tight',dpi=dpi)  
    if ShowImage:
        plt.show()
    elif mode=='FittingCurves':
        plt.close() # so, each video frame contains one curve only
    return()

#--- Main Part ---

noise_CDF='Uniform' # options:  'Normal' or 'Uniform'
sampling='Enhanced' # options: 'Enhanced' or 'Standard'
mode='ConfidenceRegion'   # options: 'ConfidenceRegion' or 'FittingCurves' 
npts = 250        # number of points in training set

ShowImage = False # set to False for video production
dpi=100    # image resolution in dpi (100 for gif / 300 for video)
flist=[]   # list of image filenames for the video
gif=[]     # used to produce the gif image
nframes=50 # number of frames in video

for frame in range(0,nframes): 

    # Global variables: dpi, frame, image 

    image='ellipse'+str(frame)+'.png' # filename of image in current frame
    print(image) # show progress on the screen

    if mode=='ConfidenceRegion':
        seed=frame # new set of random numbers for each image 
        noise=0.8   # amount of noise added to to training set
        tmin=0  #np.pi           # training set: ellipse arc starts at tmin
        tmax = 2*np.pi   # training set: ellipse arc ends at tmax
        params = 4, -3.5, 7, 1, np.pi/4 # ellipse parameters, see [*] ### 1, 1 for axis / 7 7
    elif mode=='CurveFitting':
        seed = 100        # same seed (random number generator) for all images
        p=frame/(nframes-1) # assumes nframes > 1
        noise=3*(1-p)*(1-p) # amount of noise added to to training set
        tmin=(1-p)*np.pi  # training set: ellipse arc starts at tmin
        tmax = 2*np.pi    # training set: ellipse arc ends at tmax
        params = 4, -3.5, 7, 1+6*(1-p), 2*(p+np.pi/3) # ellipse parameters, see [*]

    # call to main function    
    # description of params [*]
    #     first two is center of ellipse, last one rotation angle
    #     the two in the middle are the semi-major and semi-minor axes
    main(npts, noise, seed, tmin, tmax, params, sampling)

    # processing images for video and animated gif production (using pillow library)
    im = Image.open(image)
    if frame==0:  
      width, height = im.size  # determines the size of all future images
      width=2*int(width/2)
      height=2*int(height/2)
      fixedSize=(width,height) # even number of pixels for video production 
    im = im.resize(fixedSize)  # all images must have same size to produce video
    gif.append(im)  # to produce Gif image [uses lots of memory if dpi > 100] 
    im.save(image,"PNG") # save resized image for video production
    flist.append(image)

# output video / fps is number of frames per second
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(flist, fps=20) 
clip.write_videofile('ellipseFitting.mp4')

# output video as gif file 
gif[0].save('ellipseFitting.gif',save_all=True, append_images=gif[1:],loop=0)  


