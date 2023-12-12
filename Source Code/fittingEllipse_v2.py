# Fitting ellipse via least squares

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

    return x0, y0, ap, bp, phi

def sample_from_ellipse_even(x0, y0, ap, bp, phi, tmin, tmax, npts):

    npoints = 1000
    delta_theta=2.0*np.pi/npoints
    theta=[0.0]
    delta_s=[0.0]
    integ_delta_s=[0.0]
    integ_delta_s_val=0.0
    for iTheta in range(1,npoints+1):
        delta_s_val=np.sqrt(ap**2*np.sin(iTheta*delta_theta)**2+ \
                            bp**2*np.cos(iTheta*delta_theta)**2)
        theta.append(iTheta*delta_theta)
        delta_s.append(delta_s_val)
        integ_delta_s_val = integ_delta_s_val+delta_s_val*delta_theta
        integ_delta_s.append(integ_delta_s_val)
    integ_delta_s_norm = []
    for iEntry in integ_delta_s:
        integ_delta_s_norm.append(iEntry/integ_delta_s[-1]*2.0*np.pi)    
    
    x=[]
    y=[] 
    for k in range(npts):
        t = tmin + (tmax-tmin)*k/npts
        for lookup_index in range(len(integ_delta_s_norm)):
            lower=integ_delta_s_norm[lookup_index]
            upper=integ_delta_s_norm[lookup_index+1]
            if (t >= lower) and  (t < upper):
                t2 = theta[lookup_index]
                break    
        x.append(x0 + ap*np.cos(t2)*np.cos(phi) - bp*np.sin(t2)*np.sin(phi))
        y.append(y0 + ap*np.cos(t2)*np.sin(phi) + bp*np.sin(t2)*np.cos(phi))

    return x, y

def sample_from_ellipse(x0, y0, ap, bp, phi, tmin, tmax, npts): 

    x = np.empty(npts)
    y = np.empty(npts)
    x_unsorted = np.empty(npts)
    y_unsorted = np.empty(npts)
    angle = np.empty(npts)

    global urs, vrs

    if frame == 0:
        cov=[[ap,0],[0,bp]]
        urs, vrs = np.random.multivariate_normal([0, 0], cov, size = npts_max).T

    # sample from multivariate normal, then rescale 
    count = 0
    index = 0
    while count < npts:
        u = urs[index]
        v = vrs[index]
        index += 1
        d=np.sqrt(u*u/(ap*ap) + v*v/(bp*bp))
        u=u/d
        v=v/d
        t = np.pi + np.arctan2(-ap*v,-bp*u)   
        if t >= tmin and t <= tmax:
            x_unsorted[count] = x0 + np.cos(phi)*u - np.sin(phi)*v
            y_unsorted[count] = y0 + np.sin(phi)*u + np.cos(phi)*v
            angle[count]=t
            count=count+1

    # sort the points x, y for nice rendering with mpl.plot
    hash={}
    hash = dict(enumerate(angle.flatten(), 0)) # convert array angle to dictionary
    idx=0
    for w in sorted(hash, key=hash.get):
        x[idx]=x_unsorted[w]
        y[idx]=y_unsorted[w]
        idx=idx+1

    return x, y

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi, sampling='Standard'):

    # Return npts points on the ellipse described by the params = x0, y0, ap,
    # bp, e, phi for values of the parametric variable t between tmin and tmax.

    x0, y0, ap, bp, phi = params
    
    if sampling=='Standard':
        t = np.linspace(tmin, tmax, npts)
        x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    elif sampling=='Enhanced':
        x, y = sample_from_ellipse(x0, y0, ap, bp, phi, tmin, tmax, npts) 
    elif sampling=='Even':
        x, y = sample_from_ellipse_even(x0, y0, ap, bp, phi, tmin, tmax, npts) 

    return x, y

def vgplot(x, y, color, npts, tmin, tmax):

    plt.plot(x, y, linewidth=0.8, color=color) # plot exact ellipse 
    # fill gap (missing segment in the ellipse plot) if plotting full ellipse
    if tmax-tmin > 2*np.pi - 0.001:
        gap_x=[x[nlocs-1],x[0]]
        gap_y=[y[nlocs-1],y[0]]
        plt.plot(gap_x, gap_y, linewidth=0.8, color=color)
    plt.xticks([])  
    plt.yticks([])  
    plt.xlim(-1 + min(x), 1 + max(x)) 
    plt.ylim(-1 + min(y), 1 + max(y)) 
    return()

def main(npts, noise, seed, tmin, tmax, params, sampling):

    # params = x0, y0, ap, bp, phi (input params for ellipse)
    global ur, vr 

    # Get points x, y on the exact ellipse and plot them
    x, y = get_ellipse_pts(params, npts, tmin, tmax, sampling)

    # perturb x, y on the ellipse with some noise, to produce training set
    if frame == 0: 
      cov = [[1,0],[0,1]]  
      np.random.seed(seed)
      ur, vr = np.random.multivariate_normal([0, 0], cov, size = npts_max).T ### npts).T
    x += noise * ur[0:npts]  
    y += noise * vr[0:npts]  

    # get and print exact and estimated ellipse params
    coeffs = fit_ellipse(x, y) # get quadratic form coeffs
    print('True ellipse    :  x0, y0, ap, bp, phi = %+.5f %+.5f %+.5f %+.5f %+.5f' % params)
    fitted_params = cart_to_pol(coeffs)  # convert quadratic coeffs to params
    print('Estimated values:  x0, y0, ap, bp, phi = %+.5f %+.5f %+.5f %+.5f %+.5f' % fitted_params)
    print()

    # plot training set points in red
    plt.scatter(x, y,s = 3.5,color = 'red') 
 
    # get nlocs points on the fitted ellipse and plot them
    x, y = get_ellipse_pts(fitted_params, nlocs, tmin, tmax, sampling) 
    vgplot(x, y,'blue', nlocs, tmin, tmax)

    # save plots in a picture [filename is image]
    plt.savefig(image, bbox_inches='tight',dpi=dpi)  
    plt.close() # so, each video frame contains one curve only
    return()

#--- Main Part: Initializationa

sampling= 'Enhanced'         # options: 'Enhanced', 'Standard', 'Even' 
npts_max = 50000         # max size of random arrays
nlocs = 2500             # number of points used to represent true ellipse 

dpi =240       # image resolution in dpi (100 for gif / 300 for video)
flist = []     # list of image filenames for the video
gif = []       # used to produce the gif image
nframes = 500  # number of frames in video

# intialize plotting parameters
plt.rcParams['axes.linewidth'] = 0.8
plt.rc('axes',edgecolor='black') # border color
plt.rc('xtick', labelsize=6) # font size, x axis  
plt.rc('ytick', labelsize=6) # font size, y axis

#--- Main part: Main loop

for frame in range(0,nframes): 

    # Global variables: dpi, frame, image
    image='ellipse'+str(frame)+'.png' # filename of image in current frame
    print("Creating image",image) # show progress on the screen

    # params = (x0, y0, ap, bp, phi) : first two coeffs is center of ellipse, last one 
    # is rotation angle, the two in the middle are the semi-major and semi-minor axes.
    #
    # Also: 0 <= tmin < tmax <= 2 pi determine start / end of ellipse arc

    # parameters used for current frame 
    seed = 100            # same seed (random number generator) for all images 
    p = frame/(nframes-1) # assumes nframes > 1
    noise = (1-p)*(1-p)   # amount of noise added to to training set
    npts = int(100*(2-p)) # number of points in training set, < npts_max 
    tmin= (1-p)*np.pi     # training set: ellipse arc starts at tmin >= 0
    tmax= 2*np.pi         # training set: ellipse arc ends at tmax  < 2*Pi 
    params = 4, -3.5, 7, 1+6*(1-p), (p+np.pi/3) # ellipse parameters 

    # call to main function 
    main(npts, noise, seed, tmin, tmax, params, sampling)

    # processing images for video and animated gif production (using pillow library)
    im = Image.open(image)
    if frame==0:  
        width, height = im.size  # determines the size of all future images
        width=2*int(width/2)
        height=2*int(height/2)
        fixedSize=(width,height) # even number of pixels for video production 
    im = im.resize(fixedSize)  # all images must have same size to produce video
    gif.append(im)       # to produce Gif image [uses lots of memory if dpi > 100] 
    im.save(image,"PNG") # save resized image for video production
    flist.append(image)

# output video / fps is number of frames per second
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(flist, fps=20) 
clip.write_videofile('ellipseFitting.mp4')

# output video as gif file 
gif[0].save('ellipseFitting.gif',save_all=True, append_images=gif[1:],loop=0)  

#--- Making picture with thumbmails
#
# note: nframes must be a multiple of n_thumbnails

columns  = 5
rows = 5
n_thumbnails = columns * rows
increment = int(nframes / n_thumbnails) 

import matplotlib.image as mpimg
cnt = 1

for frame in range(0, nframes, increment):

    fname = flist[frame]
    img = mpimg.imread(fname)
    plt.subplot(rows, columns, cnt)
    plt.xticks([])  
    plt.yticks([])
    plt.axis('off')
    plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
    plt.imshow(img)
    cnt += 1

plt.show()
