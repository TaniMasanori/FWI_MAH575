# %% [markdown]
# # 2D Kernel with SPECFEM2D - Multiple Receivers
# By Andrea R.  
# Utility functions written by Ridvan Orsvuran.  
# Following file structure as in the Seisflows example (By Bryant Chow)

# %%
import os
import shutil
import matplotlib
import numpy as np
import FunctionsPlotBin
import matplotlib.pyplot as plt

from UtilityFunctions import read_trace, specfem2D_prep_save_forward, replace_line, save_trace, specfem2D_prep_adjoint, grid
from scipy.integrate import simps

# %% [markdown]
# # Domain: 
# 2D:   
# x-dir = 4000 m   
# z-dir = 4000 m   
# 
# ### Source location
# original x:    1000 m     
# original z:    2000 m  
# 
# ### Output stations locations: 
# Station #1   AAS0001                             
# original x:    3000 m  
# original z:    2000 m  
# 
# Station #2   AAS0002                             
# original x:    2000 m  
# original z:    3000 m  
# 
# Station #3   AAS0003                             
# original x:    2000 m  
# original z:    1000 m  
# 
# ### Boundary conditions 
# Type: STACEY_ABSORBING_CONDITIONS  
# absorbbottom                    = true  
# absorbright                     = true  
# absorbtop                       = true  
# absorbleft                      = true  
# 
# ### Velocity model:
# 
# #### Initial model:  
# Model: P (or PI) velocity min,max =    3000 m/s              
# Model: S velocity min,max         =    1800 m/s        
# Model: density min,max            =    2700 kg/m3           
# 
# #### True model (~1% perturbation of the Vs - initial model):   
# Model: P (or PI) velocity min,max =    3000 m/s                
# Model: S velocity min,max         =    1820 m/s          
# Model: density min,max            =    2700 kg/m3      

# %% [markdown]
# ### Set Specfem2D and work directories 

# %%
specfem2d_path = "/home/masa/" # for desktop machine
#specfem2d_path = "/home/masan/" # for laptop 
EXAMPLE = os.path.join(os.getcwd(),"Examples", "DATA_Example01")
WORKDIR = os.path.join(os.getcwd(),"work")

# Incase we've run this docs page before, delete the working directory before remaking
if os.path.exists(WORKDIR):
    shutil.rmtree(WORKDIR)
os.makedirs(WORKDIR, exist_ok=True)

# %%
# Distribute the necessary file structure of the SPECFEM2D repository that we will reference
SPECFEM2D_ORIGINAL = os.path.join(specfem2d_path, "specfem2d") 
SPECFEM2D_BIN_ORIGINAL = os.path.join(SPECFEM2D_ORIGINAL, "bin")
SPECFEM2D_DATA_ORIGINAL = os.path.join(SPECFEM2D_ORIGINAL, "DATA")

# The SPECFEM2D working directory that we will create separate from the downloaded repo
SPECFEM2D_WORKDIR = os.path.join(WORKDIR, "ExampleKernelMultiReceiver")
SPECFEM2D_BIN = os.path.join(SPECFEM2D_WORKDIR, "bin")
SPECFEM2D_DATA = os.path.join(SPECFEM2D_WORKDIR, "DATA")
SPECFEM2D_OUTPUT = os.path.join(SPECFEM2D_WORKDIR, "OUTPUT_FILES")

# Pre-defined locations of velocity models we will generate using the solver
SPECFEM2D_MODEL_INIT = os.path.join(SPECFEM2D_WORKDIR, "OUTPUT_FILES_INIT")
SPECFEM2D_MODEL_TRUE = os.path.join(SPECFEM2D_WORKDIR, "OUTPUT_FILES_TRUE")

# %%
# Copy the binary files incase we update the source code. These can also be symlinked.
shutil.copytree(SPECFEM2D_BIN_ORIGINAL, SPECFEM2D_BIN)

# Copy the DATA/ directory
shutil.copytree(EXAMPLE, SPECFEM2D_DATA)
!pwd
!ls

# %%
# Create a new STATIONS file with multiple receivers
stations_content = """S0001    AA          3000.0000000        2000.0000000       0.0         0.0
S0002    AA          2000.0000000        3000.0000000       0.0         0.0
S0003    AA          2000.0000000        1000.0000000       0.0         0.0
"""

with open(os.path.join(SPECFEM2D_DATA, "STATIONS"), "w") as f:
    f.write(stations_content)

# %% [markdown]
# ### Generate true model

# %%
os.chdir(SPECFEM2D_DATA)
specfem2D_prep_save_forward("Par_file")
# Modify the Par_file to increase Initial Vs by ~1% 
replace_line("Par_file",33,"P_SV                            = .false. \n")
replace_line("Par_file",262,'1 1 2700.d0 3000.d0 1820.d0 0 0 9999 9999 0 0 0 0 0 0 \n')

# %%
# create the OUTPUT_FILES directory before running 
os.chdir(SPECFEM2D_WORKDIR)
if os.path.exists(SPECFEM2D_OUTPUT):
    shutil.rmtree(SPECFEM2D_OUTPUT)
os.mkdir(SPECFEM2D_OUTPUT)
!ls

# %%
os.chdir(SPECFEM2D_WORKDIR)
!bin/xmeshfem2D > OUTPUT_FILES/mesher_log.txt
!bin/xspecfem2D > OUTPUT_FILES/solver_log.txt

# Move the model files (*.bin) into the OUTPUT_FILES directory
!mv DATA/*bin OUTPUT_FILES

# Make sure we don't overwrite this target model when creating our initial model in the next step
!mv OUTPUT_FILES OUTPUT_FILES_TRUE

!head OUTPUT_FILES_TRUE/solver_log.txt
!tail OUTPUT_FILES_TRUE/solver_log.txt

# %%
x_coords_file = 'OUTPUT_FILES_TRUE/proc000000_x.bin'
z_coords_file = 'OUTPUT_FILES_TRUE/proc000000_z.bin'
Vs_true       = 'OUTPUT_FILES_TRUE/proc000000_vs.bin'

# Plot 
FunctionsPlotBin.plotbin(x_coords_file,z_coords_file,Vs_true,SPECFEM2D_WORKDIR+'/Vs_true','Vs_true=m/s')

# %% [markdown]
# ### Generate initial model

# %%
os.chdir(SPECFEM2D_DATA)
replace_line("Par_file",262,'1 1 2700.d0 3000.d0 1800.d0 0 0 9999 9999 0 0 0 0 0 0 \n')

# %%
# create the OUTPUT_FILES directory before running 
os.chdir(SPECFEM2D_WORKDIR)
if os.path.exists(SPECFEM2D_OUTPUT):
    shutil.rmtree(SPECFEM2D_OUTPUT)
os.mkdir(SPECFEM2D_OUTPUT)
!ls

# %%
os.chdir(SPECFEM2D_WORKDIR)
!bin/xmeshfem2D > OUTPUT_FILES/mesher_log.txt
!bin/xspecfem2D > OUTPUT_FILES/solver_log.txt

# Move the model files (*.bin) into the OUTPUT_FILES directory
# The binary files of the velocity models are stored in DATA after running xspecfem2D
!mv DATA/*bin OUTPUT_FILES

# Store output files of initial model
!mv OUTPUT_FILES OUTPUT_FILES_INIT

!head OUTPUT_FILES_INIT/solver_log.txt
!tail OUTPUT_FILES_INIT/solver_log.txt

# %%
x_coords_file = 'OUTPUT_FILES_INIT/proc000000_x.bin'
z_coords_file = 'OUTPUT_FILES_INIT/proc000000_z.bin'
Vs_true       = 'OUTPUT_FILES_INIT/proc000000_vs.bin'

# Plot 
FunctionsPlotBin.plotbin(x_coords_file,z_coords_file,Vs_true,SPECFEM2D_WORKDIR+'/Vs_init','Vs_init=m/s')

# %% [markdown]
# ### 3. Plot synthetic seismograms for all receivers

# %%
os.chdir(SPECFEM2D_WORKDIR)
station_ids = ["S0001", "S0002", "S0003"]

fig, axes = plt.subplots(len(station_ids), 1, figsize=(20, 12), sharex=True)
matplotlib.rcParams.update({'font.size': 14})

for i, station_id in enumerate(station_ids):
    # Read synthetic seismogram
    obsd = read_trace(os.path.join("OUTPUT_FILES_TRUE",f"AA.{station_id}.BXY.semd"))
    synt = read_trace(os.path.join("OUTPUT_FILES_INIT",f"AA.{station_id}.BXY.semd"))

    # Process data
    obsd.detrend("simple")
    obsd.taper(0.05)
    obsd.filter("bandpass", freqmin=0.01, freqmax=20)

    synt.detrend("simple")
    synt.taper(0.05)
    synt.filter("bandpass", freqmin=0.01, freqmax=20)
    
    # Plot
    axes[i].plot(obsd.times()+obsd.stats.b, obsd.data, "b", label="Obsd")
    axes[i].plot(synt.times()+synt.stats.b, synt.data, "r", label="Synt")
    axes[i].set_xlim(synt.stats.b, synt.times()[-1]+synt.stats.b)
    axes[i].legend(frameon=False)
    axes[i].set_ylabel(f"Station {station_id}\nDisplacement (m)")
    axes[i].tick_params(axis='both',which='major',labelsize=14)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Comparison of Synthetic and Observed Seismograms", fontsize=16)
plt.tight_layout()

# %%
os.chdir(SPECFEM2D_WORKDIR)
station_ids = ["S0001", "S0002", "S0003"]

# 1. Cross-correlation traveltime misfit
def calculate_cc_traveltime_misfit(obsd, synt):
    # Cross correlation to find time shift
    cc = np.correlate(obsd.data, synt.data, mode='full')
    dt = obsd.stats.delta
    shift_samples = np.argmax(cc) - (len(obsd.data) - 1)
    time_shift = shift_samples * dt
    
    # Misfit calculation
    misfit = 0.5 * time_shift**2
    
    # Adjoint source calculation
    # Get the derivative of synthetic
    deriv = np.gradient(synt.data, dt)
    
    # Normalize by energy of the derivative
    norm_factor = 1.0 / np.sum(deriv**2 * dt)
    adj_source = time_shift * norm_factor * deriv
    
    return misfit, adj_source

# 2. Cross-correlation amplitude misfit
def calculate_cc_amplitude_misfit(obsd, synt):
    # Cross correlation to find amplitude difference
    cc = np.correlate(obsd.data, synt.data, mode='full')
    max_cc_idx = np.argmax(cc) - (len(obsd.data) - 1)
    
    # Shift the synthetic to align with observed
    if max_cc_idx > 0:
        synt_aligned = np.concatenate([np.zeros(max_cc_idx), synt.data[:-max_cc_idx]])
    elif max_cc_idx < 0:
        synt_aligned = np.concatenate([synt.data[-max_cc_idx:], np.zeros(-max_cc_idx)])
    else:
        synt_aligned = synt.data
    
    # Calculate amplitude ratio
    energy_obs = np.sum(obsd.data**2)
    energy_syn = np.sum(synt_aligned**2)
    
    if energy_syn > 0:
        amplitude_ratio = np.sqrt(energy_obs / energy_syn)
    else:
        amplitude_ratio = 1.0
    
    # Amplitude anomaly
    dlnA = np.log(amplitude_ratio)
    
    # Misfit
    misfit = 0.5 * dlnA**2
    
    # Adjoint source is the time-reversed synthetic scaled by the amplitude anomaly
    norm_factor = 1.0 / np.sum(synt_aligned**2 * obsd.stats.delta)
    adj_source = dlnA * norm_factor * synt_aligned
    
    return misfit, adj_source

# 3. Waveform misfit
def calculate_waveform_misfit(obsd, synt):
    # Waveform difference
    diff = synt.data - obsd.data
    
    # Misfit calculation
    dt = obsd.stats.delta
    misfit = 0.5 * np.sum(diff**2) * dt
    
    # Adjoint source is simply the difference
    adj_source = diff
    
    return misfit, adj_source

# Dictionary to store all misfit types
misfit_types = {
    "cc_traveltime": {"function": calculate_cc_traveltime_misfit, "total": 0, "adjoint_sources": {}},
    "cc_amplitude": {"function": calculate_cc_amplitude_misfit, "total": 0, "adjoint_sources": {}},
    "waveform": {"function": calculate_waveform_misfit, "total": 0, "adjoint_sources": {}}
}

# Calculate all misfit types for all stations
for station_id in station_ids:
    # Read and process seismograms
    obsd = read_trace(os.path.join("OUTPUT_FILES_TRUE", f"AA.{station_id}.BXY.semd"))
    synt = read_trace(os.path.join("OUTPUT_FILES_INIT", f"AA.{station_id}.BXY.semd"))

    # Process data
    obsd.detrend("simple")
    obsd.taper(0.05)
    obsd.filter("bandpass", freqmin=0.01, freqmax=20)

    synt.detrend("simple")
    synt.taper(0.05)
    synt.filter("bandpass", freqmin=0.01, freqmax=20)
    
    # Calculate each misfit type
    for misfit_type, misfit_data in misfit_types.items():
        station_misfit, adj_source = misfit_data["function"](obsd, synt)
        misfit_data["total"] += station_misfit
        
        # Create adjoint source trace
        adj = synt.copy()
        adj.data = adj_source
        adj.detrend("simple")
        adj.taper(0.05)
        adj.filter("bandpass", freqmin=0.01, freqmax=20)
        
        misfit_data["adjoint_sources"][station_id] = adj

# Print total misfits
for misfit_type, misfit_data in misfit_types.items():
    print(f"Total {misfit_type} misfit: {misfit_data['total']:.6e}")

# %%
# Plot all adjoint sources for each misfit type
misfit_names = {
    "cc_traveltime": "Cross-correlation Traveltime",
    "cc_amplitude": "Cross-correlation Amplitude",
    "waveform": "Waveform"
}

fig, axes = plt.subplots(len(misfit_types), len(station_ids), figsize=(20, 15), sharex=True)
matplotlib.rcParams.update({'font.size': 14})

for i, (misfit_type, misfit_data) in enumerate(misfit_types.items()):
    for j, station_id in enumerate(station_ids):
        adj = misfit_data["adjoint_sources"][station_id]
        axes[i, j].plot(adj.times()+adj.stats.b, adj.data, "g")
        axes[i, j].set_xlim(adj.stats.b, adj.times()[-1]+adj.stats.b)
        
        if i == 0:
            axes[i, j].set_title(f"Station {station_id}")
        
        if j == 0:
            axes[i, j].set_ylabel(f"{misfit_names[misfit_type]}\nAdjoint Source")

axes[-1, len(station_ids)//2].set_xlabel("Time (s)")
fig.suptitle("Adjoint Sources for Different Misfit Types", fontsize=16)
plt.tight_layout()

# %% [markdown]
# ### Run Adjoint Simulations for Different Misfit Types
# 
# Now we will run the adjoint simulation for each of the three misfit types to calculate their respective kernels.

# %%
# Function to run adjoint simulation and calculate kernel for a specific misfit type
def run_adjoint_simulation(misfit_type):
    os.chdir(SPECFEM2D_WORKDIR)
    
    # Create output directory for this misfit type
    output_dir = f"OUTPUT_FILES_ADJ_{misfit_type}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create SEM directory for adjoint sources
    sem_dir = f"SEM_{misfit_type}"
    if os.path.exists(sem_dir):
        shutil.rmtree(sem_dir)
    os.makedirs(sem_dir, exist_ok=True)
    
    # Save all adjoint sources for this misfit type
    adjoint_sources = misfit_types[misfit_type]["adjoint_sources"]
    for station_id, adj in adjoint_sources.items():
        save_trace(adj, f"{sem_dir}/AA.{station_id}.BXY.adj")
    
    # Prepare Par_file for adjoint simulation
    os.chdir(SPECFEM2D_DATA)
    specfem2D_prep_adjoint("Par_file")
    
    # Create OUTPUT_FILES directory and copy initial model results
    os.chdir(SPECFEM2D_WORKDIR)
    if os.path.exists(SPECFEM2D_OUTPUT):
        shutil.rmtree(SPECFEM2D_OUTPUT)
    os.mkdir(SPECFEM2D_OUTPUT)
    os.system(f"cp OUTPUT_FILES_INIT/* OUTPUT_FILES")
    
    # Also copy the adjoint sources from the SEM directory to the main SEM directory
    # as expected by the SPECFEM2D code
    if os.path.exists("SEM"):
        shutil.rmtree("SEM")
    os.makedirs("SEM", exist_ok=True)
    os.system(f"cp {sem_dir}/* SEM/")
    
    # Run adjoint simulation
    os.chdir(SPECFEM2D_WORKDIR)
    os.system(f"bin/xmeshfem2D > OUTPUT_FILES/mesher_log.txt")
    os.system(f"bin/xspecfem2D > OUTPUT_FILES/solver_log.txt")
    
    # Move output files to misfit-specific directory
    os.system(f"mv OUTPUT_FILES {output_dir}")
    
    print(f"Adjoint simulation for {misfit_type} completed successfully")
    
    # Load and return the kernel data
    kernel_file = f"./{output_dir}/proc000000_rhop_alpha_beta_kernel.dat"
    if os.path.exists(kernel_file):
        data = np.loadtxt(kernel_file)
        return data
    else:
        print(f"Warning: Kernel file not found for {misfit_type}")
        return None

# Run adjoint simulations for all misfit types
kernels = {}
for misfit_type in misfit_types.keys():
    print(f"Running adjoint simulation for {misfit_type}...")
    kernel_data = run_adjoint_simulation(misfit_type)
    if kernel_data is not None:
        kernels[misfit_type] = kernel_data

# %%
# Plot and compare the kernels for different misfit types
def plot_kernel(kernel_data, title, vmax=None):
    x = kernel_data[:, 0]
    z = kernel_data[:, 1]
    beta = kernel_data[:, 4]  # beta kernel (S-wave)
    
    if vmax is None:
        vmax = max(abs(np.percentile(beta, 99)), abs(np.percentile(beta, 1)))
    
    X, Z, BETA = grid(x, z, beta)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(BETA, vmax=vmax, vmin=-vmax, extent=[x.min(), x.max(), z.min(), z.max()],
                   cmap="seismic_r")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    
    # Plot source and receivers
    ax.scatter(1000, 2000, 200, marker="*", color="black", edgecolor="white", label="Source")
    ax.scatter(3000, 2000, 100, marker="v", color="blue", edgecolor="white", label="Station 1")
    ax.scatter(2000, 3000, 100, marker="v", color="green", edgecolor="white", label="Station 2")
    ax.scatter(2000, 1000, 100, marker="v", color="red", edgecolor="white", label="Station 3")
    
    plt.colorbar(im, ax=ax)
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    
    return fig, ax

# Plot all kernels with the same color scale
if kernels:
    # Find common scale for better comparison
    all_beta_values = []
    for kernel_data in kernels.values():
        all_beta_values.extend(kernel_data[:, 4])
    
    common_vmax = max(abs(np.percentile(all_beta_values, 99)), abs(np.percentile(all_beta_values, 1)))
    
    # Plot each kernel
    for misfit_type, kernel_data in kernels.items():
        title = f"S-wave Kernel for {misfit_names[misfit_type]} Misfit"
        plot_kernel(kernel_data, title, vmax=common_vmax)
        
    # Create a comparison plot
    fig, axes = plt.subplots(1, len(kernels), figsize=(18, 6))
    
    for i, (misfit_type, kernel_data) in enumerate(kernels.items()):
        x = kernel_data[:, 0]
        z = kernel_data[:, 1]
        beta = kernel_data[:, 4]
        
        X, Z, BETA = grid(x, z, beta)
        
        im = axes[i].imshow(BETA, vmax=common_vmax, vmin=-common_vmax, 
                           extent=[x.min(), x.max(), z.min(), z.max()],
                           cmap="seismic_r")
        axes[i].set_title(f"{misfit_names[misfit_type]}")
        axes[i].set_xlabel("X (m)")
        
        if i == 0:
            axes[i].set_ylabel("Z (m)")
        
        # Plot source and receivers
        axes[i].scatter(1000, 2000, 80, marker="*", color="black", edgecolor="white")
        axes[i].scatter(3000, 2000, 40, marker="v", color="blue", edgecolor="white")
        axes[i].scatter(2000, 3000, 40, marker="v", color="green", edgecolor="white")
        axes[i].scatter(2000, 1000, 40, marker="v", color="red", edgecolor="white")
    
    plt.colorbar(im, ax=axes, label="Kernel Value")
    fig.suptitle("Comparison of S-wave Kernels for Different Misfit Types", fontsize=16)
    plt.tight_layout()
else:
    print("No kernel data available for plotting")

# %% [markdown]
# ## Analysis of Kernel Types
# 
# Let's analyze and compare the three different types of misfit kernels:
# 
# 1. **Cross-correlation Traveltime Kernel**:
#    - Sensitive to the phase differences between observed and synthetic waveforms
#    - Most useful for capturing velocity structure along ray paths
#    - Shows where model updates would improve the timing of arrivals
# 
# 2. **Cross-correlation Amplitude Kernel**:
#    - Sensitive to amplitude differences between observed and synthetic waveforms
#    - Helps identify areas affecting the energy of the waves
#    - Shows where model updates would improve amplitude match
# 
# 3. **Waveform Misfit Kernel**:
#    - Sensitive to all differences between observed and synthetic waveforms
#    - Combines both phase and amplitude information
#    - Can be noisy because it tries to fit everything including complexities in the data
# 
# Different misfit types highlight different features of the subsurface, making them useful for different inversion goals.

# %%
# Compare the kernel amplitude distributions
if kernels:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for misfit_type, kernel_data in kernels.items():
        beta_values = kernel_data[:, 4]  # beta kernel (S-wave)
        
        # Plot histogram of non-zero kernel values
        ax.hist(beta_values[beta_values != 0], bins=50, alpha=0.5, 
                label=misfit_names[misfit_type])
    
    ax.set_xlabel("Kernel Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of S-wave Kernel Values")
    ax.legend()
    plt.tight_layout()
    
    # Plot kernel amplitude along a horizontal slice
    z_index = 2000  # y-coordinate to extract (m)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for misfit_type, kernel_data in kernels.items():
        x = kernel_data[:, 0]
        z = kernel_data[:, 1]
        beta = kernel_data[:, 4]
        
        # Find points closest to the desired z-coordinate
        z_slice_indices = np.abs(z - z_index).argsort()[:1000]  # Get closest 1000 points
        
        # Sort by x-coordinate
        sorted_indices = np.argsort(x[z_slice_indices])
        x_sorted = x[z_slice_indices][sorted_indices]
        beta_sorted = beta[z_slice_indices][sorted_indices]
        
        # Plot the slice
        ax.plot(x_sorted, beta_sorted, label=misfit_names[misfit_type])
    
    ax.axvline(x=1000, color='black', linestyle='--', label='Source')
    ax.axvline(x=3000, color='blue', linestyle='--', label='Station 1')
    
    ax.set_xlabel("X-coordinate (m)")
    ax.set_ylabel("Kernel Value")
    ax.set_title(f"S-wave Kernel Values Along Z = {z_index}m")
    ax.legend()
    plt.tight_layout()
    
    # Plot kernel amplitude along a vertical slice
    x_index = 2000  # x-coordinate to extract (m)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for misfit_type, kernel_data in kernels.items():
        x = kernel_data[:, 0]
        z = kernel_data[:, 1]
        beta = kernel_data[:, 4]
        
        # Find points closest to the desired x-coordinate
        x_slice_indices = np.abs(x - x_index).argsort()[:1000]  # Get closest 1000 points
        
        # Sort by z-coordinate
        sorted_indices = np.argsort(z[x_slice_indices])
        z_sorted = z[x_slice_indices][sorted_indices]
        beta_sorted = beta[x_slice_indices][sorted_indices]
        
        # Plot the slice
        ax.plot(beta_sorted, z_sorted, label=misfit_names[misfit_type])
    
    ax.axhline(y=1000, color='red', linestyle='--', label='Station 3')
    ax.axhline(y=3000, color='green', linestyle='--', label='Station 2')
    
    ax.set_xlabel("Kernel Value")
    ax.set_ylabel("Z-coordinate (m)")
    ax.set_title(f"S-wave Kernel Values Along X = {x_index}m")
    ax.legend()
    plt.tight_layout()
else:
    print("No kernel data available for analysis")

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we have:
# 
# 1. Set up a model with three receivers at different locations to compute S-wave kernels
# 2. Calculated three different types of misfit functions:
#    - Cross-correlation traveltime misfit
#    - Cross-correlation amplitude misfit
#    - Waveform misfit
# 3. Generated the corresponding adjoint sources for each misfit type
# 4. Ran adjoint simulations to compute the S-wave sensitivity kernels
# 5. Compared and analyzed the different kernel characteristics
# 
# Key observations:
# - Traveltime kernels typically have a banana-doughnut shape and are most sensitive along the raypath
# - Amplitude kernels highlight areas that control the energy of the waves
# - Waveform kernels combine both phase and amplitude information
# - Using multiple receivers creates more complex kernel patterns as sensitivities from different paths interfere
# 
# These sensitivity kernels form the foundation of FWI as they show how model changes will affect the data. Different misfit functions emphasize different aspects of wave propagation, making them suitable for different inversion scenarios.

# %%
os.chdir(SPECFEM2D_WORKDIR)
!bin/xmeshfem2D > OUTPUT_FILES/mesher_log.txt
!bin/xspecfem2D > OUTPUT_FILES/solver_log.txt

# Move the model files (*.bin) into the OUTPUT_FILES directory
# The binary files of the velocity models are stored in DATA after running xspecfem2D
!mv DATA/*bin OUTPUT_FILES

# Make sure we don't overwrite output files 
!mv OUTPUT_FILES OUTPUT_FILES_ADJ

!head OUTPUT_FILES_ADJ/solver_log.txt
!tail OUTPUT_FILES_ADJ/solver_log.txt

# %% [markdown]
# ### Plotting the Kernels from Different Misfit Types
# 
# Each misfit type (cross-correlation traveltime, cross-correlation amplitude, and waveform) has its own kernel saved in a separate output directory:
# - `/home/masa/FWI_MAH575/JupyterNotebooks/work/ExampleKernelMultiReceiver/OUTPUT_FILES_ADJ_cc_traveltime`
# - `/home/masa/FWI_MAH575/JupyterNotebooks/work/ExampleKernelMultiReceiver/OUTPUT_FILES_ADJ_cc_amplitude`
# - `/home/masa/FWI_MAH575/JupyterNotebooks/work/ExampleKernelMultiReceiver/OUTPUT_FILES_ADJ_waveform`
# 
# The kernel data is stored in `proc000000_rhop_alpha_beta_kernel.dat` files, each containing 5 columns: `x`, `z`, `rhop`, `alpha`, `beta`.

# %%
# Function to load and display kernel
def load_and_plot_kernel(kernel_path, title, vmax=None):
    if not os.path.exists(kernel_path):
        print(f"Warning: Kernel file not found at {kernel_path}")
        return None, None
    
    # Load kernel data
    data = np.loadtxt(kernel_path)
    
    # Extract columns
    x = data[:, 0]
    z = data[:, 1]
    beta = data[:, 4]  # S-wave (beta) kernel
    
    if vmax is None:
        vmax = max(abs(np.percentile(beta, 99)), abs(np.percentile(beta, 1)))
    
    # Grid the data for plotting
    X, Z, BETA = grid(x, z, beta)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(BETA, vmax=vmax, vmin=-vmax, extent=[x.min(), x.max(), z.min(), z.max()],
                  cmap="seismic_r")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    
    # Plot source and receivers
    ax.scatter(1000, 2000, 200, marker="*", color="black", edgecolor="white", label="Source")
    ax.scatter(3000, 2000, 100, marker="v", color="blue", edgecolor="white", label="Station 1")
    ax.scatter(2000, 3000, 100, marker="v", color="green", edgecolor="white", label="Station 2")
    ax.scatter(2000, 1000, 100, marker="v", color="red", edgecolor="white", label="Station 3")
    
    plt.colorbar(im, ax=ax)
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    
    return fig, data

# Paths to the kernel files for each misfit type
kernel_paths = {
    "cc_traveltime": os.path.join(os.getcwd(), "work", "ExampleKernelMultiReceiver", "OUTPUT_FILES_ADJ_cc_traveltime", "proc000000_rhop_alpha_beta_kernel.dat"),
    "cc_amplitude": os.path.join(os.getcwd(), "work", "ExampleKernelMultiReceiver", "OUTPUT_FILES_ADJ_cc_amplitude", "proc000000_rhop_alpha_beta_kernel.dat"),
    "waveform": os.path.join(os.getcwd(), "work", "ExampleKernelMultiReceiver", "OUTPUT_FILES_ADJ_waveform", "proc000000_rhop_alpha_beta_kernel.dat")
}

# Plot each kernel individually with appropriate titles
kernel_titles = {
    "cc_traveltime": "S-wave Kernel - Cross-correlation Traveltime Misfit",
    "cc_amplitude": "S-wave Kernel - Cross-correlation Amplitude Misfit",
    "waveform": "S-wave Kernel - Waveform Misfit"
}

# Store the loaded kernel data
loaded_kernels = {}

# Find a common color scale for all plots
all_beta_values = []
for misfit_type, kernel_path in kernel_paths.items():
    if os.path.exists(kernel_path):
        print(f"Loading kernel from {kernel_path}")
        data = np.loadtxt(kernel_path)
        all_beta_values.extend(data[:, 4])
        loaded_kernels[misfit_type] = data
    else:
        print(f"Kernel file not found: {kernel_path}")

if all_beta_values:
    common_vmax = max(abs(np.percentile(all_beta_values, 99)), abs(np.percentile(all_beta_values, 1)))
    print(f"Common color scale vmax = {common_vmax:.2e}")
    
    # Plot each kernel with the common scale
    for misfit_type, kernel_path in kernel_paths.items():
        if misfit_type in loaded_kernels:
            _, _ = load_and_plot_kernel(kernel_path, kernel_titles[misfit_type], vmax=common_vmax)
else:
    print("No kernel data found. Make sure to run the adjoint simulations first.")

# %%
# Create a side-by-side comparison of all three kernels
if len(loaded_kernels) > 0:
    fig, axes = plt.subplots(1, len(loaded_kernels), figsize=(20, 6))
    
    # Make sure axes is always a list/array even if there's only one kernel
    if len(loaded_kernels) == 1:
        axes = [axes]
    
    # Add each kernel to the plot
    for i, (misfit_type, data) in enumerate(loaded_kernels.items()):
        x = data[:, 0]
        z = data[:, 1]
        beta = data[:, 4]
        
        X, Z, BETA = grid(x, z, beta)
        
        im = axes[i].imshow(BETA, vmax=common_vmax, vmin=-common_vmax, 
                         extent=[x.min(), x.max(), z.min(), z.max()],
                         cmap="seismic_r")
        
        # Add title and labels
        title = misfit_type.replace('cc_', 'Cross-correlation\n').title().replace('Waveform', 'Waveform\n')
        axes[i].set_title(f"{title} Misfit")
        axes[i].set_xlabel("X (m)")
        
        if i == 0:
            axes[i].set_ylabel("Z (m)")
        
        # Add source and receiver markers
        axes[i].scatter(1000, 2000, 100, marker="*", color="black", edgecolor="white")
        axes[i].scatter(3000, 2000, 50, marker="v", color="blue", edgecolor="white")
        axes[i].scatter(2000, 3000, 50, marker="v", color="green", edgecolor="white")
        axes[i].scatter(2000, 1000, 50, marker="v", color="red", edgecolor="white")
    
    # Add a colorbar that applies to all subplots
    cbar = fig.colorbar(im, ax=axes, label="Kernel Value")
    cbar.ax.tick_params(labelsize=12)
    
    # Add a main title for the whole figure
    fig.suptitle("Comparison of S-wave Kernels for Different Misfit Types", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
else:
    print("No kernel data available for comparison")

# %% [markdown]
# ## Analysis of Different Kernel Types
# 
# The three kernel types show distinct patterns that reflect their sensitivity to different aspects of the seismic wavefield:
# 
# 1. **Cross-correlation Traveltime Kernel**:
#    - Shows a "banana-doughnut" shape between source and receivers
#    - Sensitive to velocity perturbations along the raypath
#    - Helps identify where model updates would improve arrival times
#    - Generally smoother and more focused along wave propagation paths
# 
# 2. **Cross-correlation Amplitude Kernel**:
#    - More sensitive to energy variations in the wavefield
#    - Shows where model updates would improve amplitude matching
#    - Often exhibits different spatial patterns than traveltime kernels
#    - Particularly sensitive to regions where wave amplitudes are affected
# 
# 3. **Waveform Misfit Kernel**:
#    - Combines both phase and amplitude information
#    - Often shows more complex patterns with higher frequencies
#    - May appear noisier as it tries to fit the entire waveform
#    - Useful for detailed model updates when data has high signal-to-noise ratio
# 
# Using multiple receivers creates interference patterns in the kernels, as sensitivities from different source-receiver pairs overlap. This interference can enhance or cancel out sensitivities in certain regions, which is important to understand when interpreting the kernels for FWI applications.


