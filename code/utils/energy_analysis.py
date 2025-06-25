import pandas as pd
import utils as ut
import energy_dispatch_optimizer as edo
from joblib import Parallel, delayed
from tqdm import tqdm

class EnergyAnalysis:

    def __init__(self, net_load_data):
        """
        Initialize the energy analysis with the given net_load data.

        Args:
            net_load_data: an xarray dataarray, with only time and country as coordinates
        """
        self.network = Network()
        df = net_load_data.to_pandas()
        df.columns = range(df.shape[1])
        # If there's a name on the columns (like 'time'), remove it
        df.columns.name = None
        # Rename the index
        df.index.name = "node"
        # Replace full country names with codes
        country_to_code = ut.country_name_to_country_code(None)
        df.rename(index=country_to_code, inplace=True)
        self.net_load = df
            

class Network:

    def __init__(self):
        nodes = pd.read_csv('../../inputs/set_nodes.csv')
        self.nodes = [row['node'] for i, row in nodes.iterrows()]
        self.edges_all = pd.read_csv('../../inputs/set_edges.csv')
        self.edges = self.get_single_edges()
        self.capacity_all = pd.read_csv('../../inputs/capacity_existing_power_line.csv')
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
            capacity[edge] = line_capacity
        return capacity


def get_transmission_effect(analysis):
    # initialize the optimizer based on network data
    optimizer = edo.EnergyDispatchOptimizer(analysis.network)
    # This will fill the metric_dispatch DataFrame with the results of the optimization
    metric_dispatch = pd.DataFrame(index=analysis.network.nodes,columns=analysis.net_load.columns)
    def solve_one(i, row):
        row_rel = -row
        results = optimizer.optimize_dispatch(row_rel.to_dict())
        return i, results['B']
    
    rows = list(analysis.net_load.T.iterrows())
    results = Parallel(n_jobs=-1)(delayed(solve_one)(i, row) for i, row in tqdm(rows))
    
    # Assign results back
    for i, b in results:
        metric_dispatch[i] = pd.Series(b)
    return metric_dispatch.sum()




