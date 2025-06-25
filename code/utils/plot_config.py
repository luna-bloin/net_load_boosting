import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

### Matplotlib parameters
plt.rcParams.update({
    'axes.linewidth': 0.5,
    "font.size":10,
    'lines.linewidth': 0.75,
    'lines.markersize':  3,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    "figure.dpi": 600,
    "savefig.dpi": 600
})

colors = [(196/256, 121/256, 0/256),"k",(178/256, 178/256, 178/256), (149/256, 27/256, 30/256), ( 112/256, 160/256, 205/256), (0/256, 79/256, 0/256),(0/256, 52/256, 102/256),(231/256,29/256,37/256),(0/256,173/256,207/256)] #IPCC color scheme
color2_with_alpha = to_rgba(colors[2], alpha=0.5)

#in/out paths
path = '/net/xenon/climphys/lbloin/eng_boost/'
out_path = '../../figs/'

# Function to configure the grid
def set_grid(ax=None):
    """
    Apply consistent grid styling to the specified Axes object
    """
    ax.grid(linestyle='dotted', linewidth=0.3)


def convert_ticklabels_to_strings(f,only_y=False,scientificx=False,scientificy=False):
    """
    Convert xticklabels and yticklabels to strings for each subplot in figure f.
    
    Parameters:
    f (matplotlib.figure.Figure): The figure containing subplots.
    """
                
    
    for ax in f.get_axes():  # Iterate over all axes in the figure
        if only_y == False:
            if ax.get_xticklabels():  # Check if xticklabels exist
                ax.set_xticklabels([f"10$^{{{int(np.log10(label))}}}$" if scientificx == True and label > 0 else f"{label:.6g}" for label in ax.get_xticks()])
        if ax.get_yticklabels():  # Check if yticklabels exist
            ax.set_yticklabels([f"10$^{{{int(np.log10(label))}}}$" if scientificy == True and label > 0 else f"{label:.6g}" for label in ax.get_yticks()])

def convert_colorbar_ticks_to_strings(cbar):
    """
    Convert colorbar tick labels to to strings for each subplot in figure f.
    
    Parameters:
    cbar (matplotlib.colorbar.Colorbar): The colorbar whose ticks need formatting.
    """
    ticks = cbar.get_ticks()
    formatted_ticks = [f"{tick:.6g}" for tick in ticks]
    cbar.set_ticks(ticks)  # Ensure the ticks remain the same
    cbar.set_ticklabels(formatted_ticks)  # Apply formatted labels