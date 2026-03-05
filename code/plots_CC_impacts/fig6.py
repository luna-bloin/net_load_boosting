import sys
sys.path.append("../utils/")
import utils as ut
import preprocess as pc
import plot_config as pco
import matplotlib.pyplot as plt
import string


types_of_demand={
    "IRON":"Metals",
    "CHEM":'Chemicals',
    "PAPER":'Paper',
    "FOOD":'Food',
    "exogenous":'Appliances',
    "flat":"Flat", # types of demand that have no yearly profile
    "D_RoadCar": "Trains", # profile for vehicles that need electricity to run
    "CarPark": "Charging of vehicles",# profile for vehicles that need electricity to charge
}

# === Open files ===
wi_demand = pc.open_demand_profiles()
wi_demand=wi_demand.assign_coords(
    demand_type=("demand_type", [types_of_demand[name] for name in wi_demand.demand_type.values])
)

# === Appendix figure 1 ===
f,axs=plt.subplots(2,4,figsize=(7.2,4),sharey=True)
for n, tim in enumerate(["time.dayofyear","time.hour"]):
    for x,typ in enumerate([["Trains","Charging of vehicles"],['Appliances','Flat'],["Metals","Chemicals"],['Paper','Food']]):
        to_plot = wi_demand.sel(demand_type=typ).groupby(tim).mean()
        ax = axs[n][x]
        ax.text(0.02,0.9,string.ascii_lowercase[n*4+x%4],weight="bold",transform=ax.transAxes)
        for i,to in enumerate(to_plot):
            to.plot(ax=ax,color=pco.colors[i+x*2],label=to.demand_type.values)
            ax.set_title("")
            ax.set_ylabel("")
            if tim =="time.dayofyear":
                ax.set_xlabel("")
                ax.set_xticks([60,152,244,335],labels=["Mar","Jun","Sep","Dec"])
            else:
                ax.set_xlabel("Hour of day")
    axs[n][0].set_ylabel("Normalized demand")
hands=[]
labs = []
for x in range(4):
    handles, labels = axs[0][x].get_legend_handles_labels()
    hands.extend(handles)
    labs.extend(labels)
f.legend(hands, labs, loc='lower center',ncol=4)
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
f.savefig(f"../../figs_CC_impacts/appendix_fig1.png",dpi=600,bbox_inches="tight",transparent=True)