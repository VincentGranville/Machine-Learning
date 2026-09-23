import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  
import matplotlib as mpl

def simulate_fractional_rectangular_sheet(nx, ny, H1, H2, rho):

    # Simulates a nx x ny Fractional Brownian Sheet using an FFT spectral filter.
    
    # Parameters:
    #    nx, ny: Grid dimensions (1000, 2000)
    #    H1: Hurst parameter for t1 axis (0.0 < H1 < 1.0). Controls roughness vertically.
    #    H2: Hurst parameter for t2 axis (0.0 < H2 < 1.0). Controls roughness horizontally.
    #    Low H1 or H2 means more jagged
    #    rho: Axis cross-correlation factor (-1.0 to 1.0).
  
    # 1. Generate frequencies in the Fourier domain
    fx = np.fft.fftfreq(nx).reshape(-1, 1)
    fy = np.fft.fftfreq(ny).reshape(1, -1)
    
    # Avoid division by zero at the origin frequency
    fx[0, 0] = 1e-8
    fy[0, 0] = 1e-8
    
    # 2. Apply Fractional power-law scaling filters for both directions
    # The exponent -(H + 0.5) is what mathematically dictates path roughness
    filter_x = 1.0 / (np.abs(fx) ** (H1 + 0.5))
    filter_y = 1.0 / (np.abs(fy) ** (H2 + 0.5))
    
    # 3. Create independent orthogonal noise blocks in the frequency domain
    noise_real_A = np.random.normal(0, 1, (nx, ny))
    noise_imag_A = np.random.normal(0, 1, (nx, ny))
    noise_A = noise_real_A + 1j * noise_imag_A
    
    noise_real_B = np.random.normal(0, 1, (ny, nx))
    noise_imag_B = np.random.normal(0, 1, (ny, nx))
    noise_B = (noise_real_B + 1j * noise_imag_B).T
    
    # 4. Filter the noise fields (Tensor-product convolution in Fourier space)
    field_A = noise_A * filter_x * filter_y
    field_B = noise_B * filter_x * filter_y
    
    # 5. Transform back to spatial domain using Inverse FFT
    surface_1 = np.fft.ifft2(field_A).real
    surface_2 = np.fft.ifft2(field_B).real
    
    # 6. Apply your cross-axis correlation mixing factor (rho)
    sheet = np.sqrt(1 - np.abs(rho)) * surface_1 + np.sign(rho) * np.sqrt(np.abs(rho)) * surface_2
    
    # Standardize scale so it mimics starting near 0
    sheet = sheet - sheet[0, 0]
    return sheet / np.std(sheet)


def plot_warped_sheet_3d(sheet, H1, H2):

    # 1. Downsample the sheet to a target of ~250x500 resolution
    nx, ny = sheet.shape
    stride_x = max(1, nx // 250)
    stride_y = max(1, ny // 500)
    Z_small = sheet[::stride_x, ::stride_y]
    
    # Get the downsampled dimensions
    nx_small, ny_small = Z_small.shape  # e.g., 250, 500
    
    # 2. FIX: Generate lengths using the correct respective small dimension limits
    # x maps to the column length (ny_small), y maps to row length (nx_small)
    x = np.linspace(0, 10, ny_small)  
    y = np.linspace(0, 10, nx_small)  
    X, Y = np.meshgrid(x, y)          # This safely generates a (250, 500) grid layout
    
    # 3. Apply the curving distortion equations
    X_curvy = X + 1.2 * np.sin(Y * 0.8)
    Y_curvy = Y + 1.2 * np.cos(X * 0.8)
    
    # 4. Define colormap
    colors_list = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", colors_list, N=100)

    colors_list = [(0.0, 0.0, 0.0), (0.0, 0.5, 1.0)]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", colors_list, N=500)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection='3d')
    
    # 5. FIX: Pass Z_small directly (do NOT transpose it here, as it matches X_curvy/Y_curvy shapes perfectly)
    surf = ax.plot_surface(X_curvy, Y_curvy, Z_small, cmap=custom_cmap, rstride=8, 
                           cstride=8, linewidth=0, antialiased=True)        
    ax.set_box_aspect((3, 2, 1.00))
    ax.dist = 8 
    fig.subplots_adjust(left=-0.05, right=1.05, bottom=-0.05, top=1.05)
    plt.show()


def plot_fractional_sheet_3d(sheet, H1, H2):
   
    Z = sheet  
    nx, ny = Z.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    colors_list = [(0.0, 0.0, 0.0), (0.0, 0.5, 1.0)]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", colors_list, N=500)
    #terrain_colors = mpl.colormaps['terrain'](np.linspace(0, 1, 256))
    #darkening_factor = 0.65 ## 0.55
    #terrain_colors[:, :3] *= darkening_factor
    #custom_cmap = mcolors.ListedColormap(terrain_colors).resampled(100)

    fig = plt.figure(figsize=(11, 7), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    surf = ax.plot_surface(X, Y, Z.T, rstride=8, cstride=8, cmap=custom_cmap, linewidth=0, antialiased=True)
    
    # Hide numeric values and tick lines but keep grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis='both', which='both', colors='none')
    #ax.grid(True)
    ax.set_box_aspect((3, 2, 1.00))
    ax.dist = 8 

    #fig.colorbar(surf, shrink=0.5, aspect=15)
    fig.subplots_adjust(left=-0.1, right=1.1, bottom=-0.1, top=1.1)
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    
    NX, NY = 3000, 3000
    np.random.seed(58)  # 58
    axis_correlation = -0.10
    
    # Try changing these! 
    # 0.1 structure = hyper-chaotic static noise
    # 0.5 structure = traditional Brownian motion
    # 0.8 structure = smooth rolling hills
    HURST_T1 = 0.75  
    HURST_T2 = 0.75  
    
    fractional_surface = simulate_fractional_rectangular_sheet(NX, NY, HURST_T1, HURST_T2, axis_correlation)    
    plot_fractional_sheet_3d(fractional_surface, HURST_T1, HURST_T2)
    # plot_warped_sheet_3d(fractional_surface, HURST_T1, HURST_T2)



