import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip  # to produce mp4 video
from PIL import Image  

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


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):

    # Return npts points on the ellipse described by the params = x0, y0, ap,
    # bp, e, phi for values of the parametric variable t between tmin and tmax.

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

def main(image, resolution, npts, noise, seed, tmin, tmax, params):
    # Test the algorithm with an example elliptical arc.
    x0, y0, ap, bp, phi = params  ##### ????? e ?????
    # Get some points on the ellipse (no need to specify the eccentricity).
    x, y = get_ellipse_pts((x0, y0, ap, bp, None, phi), npts, tmin, tmax)
    np.random.seed(seed)
    x += noise * np.random.normal(size=npts) 
    y += noise * np.random.normal(size=npts)

    coeffs = fit_ellipse(x, y)
    phi2 = phi % np.pi
    print('Exact  x0, y0, ap, bp, phi = : %+.5f %+.5f %+.5f %+.5f %+.5f' % (x0,y0,ap,bp,phi2))
    x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
    phi2 = phi % np.pi    
    print('Fitted x0, y0, ap, bp, phi = : %+.5f %+.5f %+.5f %+.5f %+.5f' % (x0,y0,ap,bp,phi2))

    plt.rcParams['axes.linewidth'] = 0.5
    plt.rc('axes',edgecolor='black') # border color
    plt.rc('xtick', labelsize=6) # font size, x axis 
    plt.rc('ytick', labelsize=6) # font size, y axis

    plt.scatter(x, y,s=0.5,color='red')     # given points
    x, y = get_ellipse_pts((x0, y0, ap, bp, e, phi))
    
    plt.plot(x, y, linewidth=0.5, color='blue')
    plt.savefig(image, bbox_inches='tight',dpi=resolution)  
    if ShowImage:
        plt.show()
    else:
        plt.close()
    return()

#--- main part

resolution=300    # image resolution in dpi
seed = 100        # same seed for each curve fitting
npts = 250        # number of points in training set
ShowImage = False # set to False for video production

flist=[]
gif=[] 
nframes=250

for frame in range(0,nframes): 
    image='ellipse'+str(frame)+'.png'
    p=frame/(nframes-1)
    noise=3*(1-p)*(1-p)
    tmin=(1-p)*np.pi      # training set: ellipse arc start at tim
    tmax = 2*np.pi        # training set: ellipse arc ends at tim
    params = 4, -3.5, 7, 1+6*(1-p), 2*(p+np.pi/3) # ellipse parameters
    print(image) # filename of the image
    main(image, resolution, npts, noise, seed, tmin, tmax, params)
    im = Image.open(image)
    if frame==0:  
      width, height = im.size
      width=2*int(width/2)
      height=2*int(height/2)
      fixedSize=(width,height)
    im = im.resize(fixedSize) 
    gif.append(im)  # to produce Gif image [uses lots of memory] 
    im.save(image,"PNG")
    flist.append(image)

# output video
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(flist, fps=20) 
clip.write_videofile('ellipseFitting.mp4')

# output video as gif file 
gif[0].save('ellipse.gif',save_all=True, append_images=gif[1:],loop=0)  

