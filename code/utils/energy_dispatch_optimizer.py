import numpy as np
from scipy.optimize import linprog
import networkx as nx
from typing import Dict, List, Tuple, Optional

class EnergyDispatchOptimizer:
    """
    Optimization model for renewable energy dispatch across interconnected nodes.

    Minimizes total backup generation while respecting transmission constraints.
    Based on the approach from Wohland et al. (2017) - Earth System Dynamics.
    """

    def __init__(self, network):
        """
        Initialize the optimization model.

        Parameters:
        -----------
        nodes : List[str]
            List of node identifiers (e.g., country codes)
        edges : List[Tuple[str, str]]
            List of transmission line connections (undirected)
        transmission_capacities : Dict[Tuple[str, str], float]
            Transmission capacity limits for each edge (bidirectional)
        """
        self.nodes = network.nodes
        self.edges = network.edges
        self.transmission_capacities = network.capacity
        self.n_nodes = len(network.nodes)
        self.n_edges = len(network.edges)

        # Create node index mapping
        self.node_to_idx = {node: i for i, node in enumerate(network.nodes)}

        # Create incidence matrix for flow conservation
        self.incidence_matrix = self._build_incidence_matrix()

    def _build_incidence_matrix(self) -> np.ndarray:
        """
        Build incidence matrix for network flow conservation.
        Matrix A where A[i,j] = 1 if edge j leaves node i, -1 if enters, 0 otherwise.
        """
        A = np.zeros((self.n_nodes, self.n_edges))
        for edge_idx, (node_from, node_to) in enumerate(self.edges):
            from_idx = self.node_to_idx[node_from]
            to_idx = self.node_to_idx[node_to]
            # Flow leaving node_from is positive, entering node_to is negative
            A[from_idx, edge_idx] = 1
            A[to_idx, edge_idx] = -1
        return A

    def optimize_dispatch(self, mismatches: Dict[str, float]) -> Dict[str, any]:
        """
        Solve the dispatch optimization problem for a single time step.

        Parameters:
        -----------
        mismatches : Dict[str, float]
            Energy mismatch for each node (M_i = Production_i - Demand_i)
            Positive = surplus, Negative = deficit

        Returns:
        --------
        Dict containing:
            - 'backup_generation': Dict[str, float] - Backup power by node
            - 'curtailment': Dict[str, float] - Curtailed power by node
            - 'flows': Dict[Tuple[str, str], float] - Power flows on edges
            - 'total_backup': float - Total backup generation
            - 'status': str - Optimization status
        """

        # Convert mismatches to array
        M = np.array([mismatches[node] for node in self.nodes])

        # Decision variables: [flows, backup_generation, curtailment]
        # flows: one variable per edge (can be positive or negative)
        # backup_generation: one variable per node (>= 0)
        # curtailment: one variable per node (>= 0)
        n_vars = self.n_edges + 2 * self.n_nodes

        # Objective function: minimize sum of backup generation
        # Only backup generation variables have non-zero coefficients
        c = np.zeros(n_vars)
        c[self.n_edges:self.n_edges + self.n_nodes] = 1  # Backup generation coefficients

        # Inequality constraints: transmission capacity limits
        A_ub = []
        b_ub = []

        # For each edge, add constraints: -capacity <= flow <= capacity
        for edge_idx, edge in enumerate(self.edges):
            capacity = self.transmission_capacities[edge]

            # flow <= capacity
            constraint_pos = np.zeros(n_vars)
            constraint_pos[edge_idx] = 1
            A_ub.append(constraint_pos)
            b_ub.append(capacity)

            # -flow <= capacity (equivalent to flow >= -capacity)
            constraint_neg = np.zeros(n_vars)
            constraint_neg[edge_idx] = -1
            A_ub.append(constraint_neg)
            b_ub.append(capacity)

        # Equality constraints: energy balance at each node
        # M_i + net_transmission_i + backup_i = curtailment_i
        # Rearranged: net_transmission_i + backup_i - curtailment_i = -M_i
        A_eq = np.zeros((self.n_nodes, n_vars))
        b_eq = -M  # Right-hand side

        for node_idx in range(self.n_nodes):
            # Net transmission (from incidence matrix)
            A_eq[node_idx, :self.n_edges] = -self.incidence_matrix[node_idx, :]

            # Backup generation
            A_eq[node_idx, self.n_edges + node_idx] = 1

            # Curtailment (negative because we moved it to left side)
            A_eq[node_idx, self.n_edges + self.n_nodes + node_idx] = -1

        # Variable bounds
        # Flows: unbounded (handled by capacity constraints)
        # Backup generation: >= 0
        # Curtailment: >= 0
        bounds = []

        # Flow bounds (we use capacity constraints instead of bounds for clarity)
        for _ in range(self.n_edges):
            bounds.append((None, None))

        # Backup generation bounds
        for _ in range(self.n_nodes):
            bounds.append((0, None))

        # Curtailment bounds
        for _ in range(self.n_nodes):
            bounds.append((0, None))

        # Solve optimization
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')

        if result.success:
            # Extract results
            flows = result.x[:self.n_edges]
            backup = result.x[self.n_edges:self.n_edges + self.n_nodes]
            curtailment = result.x[self.n_edges + self.n_nodes:]

            # Format results
            flow_dict = {}
            for edge_idx, edge in enumerate(self.edges):
                flow_dict[edge] = flows[edge_idx]

            backup_dict = {self.nodes[i]: backup[i] for i in range(self.n_nodes)}
            curtailment_dict = {self.nodes[i]: curtailment[i] for i in range(self.n_nodes)}

            return {
                'B': backup_dict,
                'C': curtailment_dict,
                'flows': flow_dict,
                'total_backup': result.fun,
                'status': 'optimal',
                'objective_value': result.fun
            }
        else:
            return {
                'B': {},
                'B': {},
                'flows': {},
                'total_backup': float('inf'),
                'status': f'failed: {result.message}',
                'objective_value': float('inf')
            }



    # different method for min-max optimization
    def optimize_dispatch_min_max(self, mismatches: Dict[str, float],
                                  demands: Dict[str, float]) -> Dict[str, any]:
        """
        Solve the min-max dispatch optimization problem for a single time step.

        Objective: Minimize the maximum relative backup generation (backup/demand)
        across all nodes.

        Parameters:
        -----------
        mismatches : Dict[str, float]
            Energy mismatch for each node (M_i = Production_i - Demand_i)
            Positive = surplus, Negative = deficit
        demands : Dict[str, float]
            Energy demand for each node (needed for relative backup calculation)

        Returns:
        --------
        Dict containing:
            - 'backup_generation': Dict[str, float] - Backup power by node
            - 'curtailment': Dict[str, float] - Curtailed power by node
            - 'flows': Dict[Tuple[str, str], float] - Power flows on edges
            - 'relative_backup': Dict[str, float] - Relative backup by node
            - 'total_backup': float - Total backup generation
            - 'max_relative_backup': float - Maximum relative backup ratio
            - 'status': str - Optimization status
        """

        # Convert inputs to arrays
        M = np.array([mismatches[node] for node in self.nodes])
        D = np.array([demands[node] for node in self.nodes])

        # Check for zero or negative demands (would cause division by zero)
        if np.any(D <= 0):
            zero_demand_nodes = [self.nodes[i] for i in range(self.n_nodes) if D[i] <= 0]
            raise ValueError(f"Nodes with zero or negative demand: {zero_demand_nodes}")

        # Decision variables: [flows, backup_generation, curtailment, max_relative_backup]
        # We add one auxiliary variable to represent the maximum relative backup
        n_vars = self.n_edges + 2 * self.n_nodes + 1
        max_rel_backup_idx = n_vars - 1

        # Objective function: minimize the auxiliary variable (max relative backup)
        c = np.zeros(n_vars)
        c[max_rel_backup_idx] = 1

        # Inequality constraints
        A_ub = []
        b_ub = []

        # 1. Transmission capacity limits: -capacity <= flow <= capacity
        for edge_idx, edge in enumerate(self.edges):
            capacity = self.transmission_capacities[edge]

            # flow <= capacity
            constraint_pos = np.zeros(n_vars)
            constraint_pos[edge_idx] = 1
            A_ub.append(constraint_pos)
            b_ub.append(capacity)

            # -flow <= capacity (equivalent to flow >= -capacity)
            constraint_neg = np.zeros(n_vars)
            constraint_neg[edge_idx] = -1
            A_ub.append(constraint_neg)
            b_ub.append(capacity)

        # 2. Min-max constraints: backup_i <= max_relative_backup * demand_i
        # Rearranged as: backup_i - max_relative_backup * demand_i <= 0
        for node_idx in range(self.n_nodes):
            constraint = np.zeros(n_vars)
            constraint[self.n_edges + node_idx] = 1  # backup_i coefficient
            constraint[max_rel_backup_idx] = -D[node_idx]  # -demand_i coefficient
            A_ub.append(constraint)
            b_ub.append(0)

        # Equality constraints: energy balance at each node
        # M_i + net_transmission_i + backup_i = curtailment_i
        # Rearranged: net_transmission_i + backup_i - curtailment_i = -M_i
        A_eq = np.zeros((self.n_nodes, n_vars))
        b_eq = -M  # Right-hand side

        for node_idx in range(self.n_nodes):
            # Net transmission (from incidence matrix)
            A_eq[node_idx, :self.n_edges] = -self.incidence_matrix[node_idx, :]

            # Backup generation
            A_eq[node_idx, self.n_edges + node_idx] = 1

            # Curtailment (negative because we moved it to left side)
            A_eq[node_idx, self.n_edges + self.n_nodes + node_idx] = -1

        # Variable bounds
        bounds = []

        # Flow bounds (unbounded, handled by capacity constraints)
        for _ in range(self.n_edges):
            bounds.append((None, None))

        # Backup generation bounds (>= 0)
        for _ in range(self.n_nodes):
            bounds.append((0, None))

        # Curtailment bounds (>= 0)
        for _ in range(self.n_nodes):
            bounds.append((0, None))

        # Max relative backup bound (>= 0)
        bounds.append((0, None))

        # Solve optimization
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')

        if result.success:
            # Extract results
            flows = result.x[:self.n_edges]
            backup = result.x[self.n_edges:self.n_edges + self.n_nodes]
            curtailment = result.x[self.n_edges + self.n_nodes:self.n_edges + 2 * self.n_nodes]
            max_rel_backup_opt = result.x[max_rel_backup_idx]

            # Calculate relative backup ratios
            relative_backup = backup / D
            max_relative_backup = np.max(relative_backup)

            # Format results
            flow_dict = {}
            for edge_idx, edge in enumerate(self.edges):
                flow_dict[edge] = flows[edge_idx]

            backup_dict = {self.nodes[i]: backup[i] for i in range(self.n_nodes)}
            curtailment_dict = {self.nodes[i]: curtailment[i] for i in range(self.n_nodes)}
            relative_backup_dict = {self.nodes[i]: relative_backup[i] for i in range(self.n_nodes)}

            return {
                'B': backup_dict,
                'C': curtailment_dict,
                'flows': flow_dict,
                'relative_backup': relative_backup_dict,
                'total_backup': np.sum(backup),
                'max_relative_backup': max_relative_backup,
                'max_relative_backup_optimal': max_rel_backup_opt,
                'status': 'optimal',
                'objective_value': result.fun
            }
        else:
            return {
                'B': {},
                'C': {},
                'flows': {},
                'relative_backup': {},
                'total_backup': float('inf'),
                'max_relative_backup': float('inf'),
                'max_relative_backup_optimal': float('inf'),
                'status': f'failed: {result.message}',
                'objective_value': float('inf')
            }