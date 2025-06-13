import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import glob
import utils as ut

in_path = "/net/xenon/climphys/lbloin/energy_boost/"
class EnergyAnalysis:

    def __init__(self, year,member,capacity_scenario,period):
        """
        Initialize the energy analysis with the given energy system model and scenario.

        Args:
            energy_system: Energy system model (s in the original code)
            scenario_name: Name of the scenario to analyze
            rolling_window: Size of the rolling window for time series (default: 7*24)
        """

        # self.get_year()
        self.network = Network()
        # Renewable technologies to consider
        self.technologies = ["PV","Wind_onshore","Wind_offshore","hydro_ror"]
        self.demand_techs = ["heating-demand","cooling-demand"]

        # self.capacity contains the average capacity of renewable technologies, from the average over the 60 ZEN-garden systems, based on future SSP370 climate scenario
        # self.capacity = pd.read_csv(f'{in_path}capacity_{capacity_scenario}.csv', index_col=[0,1,2])
        # self.production is pandas DataFrame with index as country codes (e.g. 'DE', 'FR') and columns as timesteps (from 0 to 8759). It includes hourly energy produced from ['photovoltaics', 'wind_onshore', 'wind_offshore', 'run-of-river_hydro']
        # self.production = pd.read_csv(f'{in_path}generation_{year}_{member}_{capacity_scenario}.csv', index_col=[0])
        # self.demand is pandas DataFrame with index as country codes (e.g. 'DE', 'FR') and columns as timesteps (from 0 to 8759). It combines electricity demand, and electricity demand for heating.
        # self.demand = pd.read_csv(f'{in_path}demand_{year}_{member}_{capacity_scenario}.csv', index_col=[0])
        file = glob.glob(f'{in_path}net_load{year}_{member}_{capacity_scenario}.csv')
        if file != []:
            self.net_load = pd.read_csv(file[0], index_col=[0])
        else:
            net_load_data = xr.open_dataset(f"{in_path}net_load_adjusted_{period}.nc").sel(member=member,capacity_scenario=capacity_scenario,time=year).convert_calendar("noleap").net_load_adjusted
            df = net_load_data.to_pandas()
            df.columns = range(df.shape[1])
            # If there's a name on the columns (like 'time'), remove it
            df.columns.name = None
            # Step 1: Rename the index
            df.index.name = "node"
            # Step 2: Replace full country names with codes
            country_to_code = ut.country_name_to_country_code(None)
            df.rename(index=country_to_code, inplace=True)
            df.to_csv(f'{in_path}net_load{year}_{member}_{capacity_scenario}.csv')
            self.net_load = df
        self.dt = pd.date_range(start=f'{year}-01-01', periods=8760, freq='h')




    def plot_net_cost_rolling(self,ws = [6, 24, 7 * 24, 28 * 24, 72 * 24, 72 * 2 * 24]):
        fig, axs = plt.subplots(2, 1, figsize=(20, 14), dpi=300)

        ax = axs[0]
        net_sum = -self.net.sum()

        for w in ws:
            # Use circular rolling mean instead of pandas' default
            rolled = circular_rolling_mean(net_sum, window=w)
            ax.plot(self.dt, rolled, label=f'{w} h', alpha=0.8)
        ax.set_title(f'Net load of {self.year}')
        ax.legend()
        ax.grid()
        ax.set_xlim(self.dt[0], self.dt[-1])

        ax = axs[1]
        cost_sum = self.cost_sum[0]

        for w in ws:
            # Use circular rolling mean instead of pandas' default
            rolled = circular_rolling_mean(cost_sum, window=w)
            ax.plot(self.dt, rolled, label=f'{w} h', alpha=0.8)
        ax.set_title(f'Dual of {self.year}')
        ax.legend()
        ax.grid()
        ax.set_xlim(self.dt[0], self.dt[-1])

        plt.tight_layout()
        plt.show()


class Network:

    def __init__(self):
        nodes = pd.read_csv('../inputs/set_nodes.csv')
        self.nodes = [row['node'] for i, row in nodes.iterrows()]
        self.edges_all = pd.read_csv('../inputs/set_edges.csv')
        self.edges = self.get_single_edges()
        self.capacity_all = pd.read_csv('../inputs/capacity_existing_power_line.csv')
        self.capacity = self.get_single_capacity()

    def get_single_edges(self):
        edges_all = self.edges_all.copy()
        edges_all['sorted_edge'] = edges_all.apply(lambda row: '-'.join(sorted([row['node_from'], row['node_to']])),axis=1)
        single_edges = edges_all.drop_duplicates(subset='sorted_edge').drop(columns='sorted_edge')
        single_edges = [(row['node_from'], row['node_to']) for i, row in single_edges.iterrows()]
        return single_edges

    def get_single_capacity(self):
        capacity = {}
        for edge in self.edges:
            node_from = edge[0]
            node_to = edge[1]
            line_capacity = max(self.capacity_all[self.capacity_all['edge'] == f'{node_to}-{node_from}']['capacity_existing'].values, self.capacity_all[self.capacity_all['edge'] == f'{node_from}-{node_to}']['capacity_existing'].values)[0]
            capacity[edge] = line_capacity#*100000
        return capacity




