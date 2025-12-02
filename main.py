import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pulp import *
import random
from typing import List, Dict, Tuple
import json
import os

os.makedirs('results/plots', exist_ok=True)


class CloudDomain:
    
    def __init__(self, domain_id: str, name: str, domain_type: str, 
                 cpu: int, memory: int, storage: int):
        self.id = domain_id
        self.name = name
        self.type = domain_type  # 'edge' or 'regional'
        self.cpu_capacity = cpu
        self.memory_capacity = memory
        self.storage_capacity = storage
        self.cpu_used = 0
        self.memory_used = 0
        self.storage_used = 0
        self.hosted_vnfs = []
        
    def can_host(self, vnf: 'VNF') -> bool:
        return (self.cpu_capacity - self.cpu_used >= vnf.cpu_required and
                self.memory_capacity - self.memory_used >= vnf.memory_required and
                self.storage_capacity - self.storage_used >= vnf.storage_required)
    
    def allocate_vnf(self, vnf: 'VNF') -> bool:
        if self.can_host(vnf):
            self.cpu_used += vnf.cpu_required
            self.memory_used += vnf.memory_required
            self.storage_used += vnf.storage_required
            self.hosted_vnfs.append(vnf)
            return True
        return False
    
    def get_utilization(self) -> Dict[str, float]:
        return {
            'cpu': (self.cpu_used / self.cpu_capacity) * 100,
            'memory': (self.memory_used / self.memory_capacity) * 100,
            'storage': (self.storage_used / self.storage_capacity) * 100
        }
    
    def __repr__(self):
        return f"CloudDomain({self.name}, Type: {self.type}, CPU: {self.cpu_used}/{self.cpu_capacity})"


class NetworkLink:
    
    def __init__(self, source: str, destination: str, 
                 bandwidth: int, latency: int):
        self.source = source
        self.destination = destination
        self.bandwidth_capacity = bandwidth
        self.latency = latency
        self.bandwidth_used = 0
        
    def has_capacity(self, required_bandwidth: int) -> bool:
        return self.bandwidth_capacity - self.bandwidth_used >= required_bandwidth
    
    def allocate_bandwidth(self, bandwidth: int) -> bool:
        if self.has_capacity(bandwidth):
            self.bandwidth_used += bandwidth
            return True
        return False


class VNF:
    
    def __init__(self, vnf_id: str, vnf_type: str):
        self.id = vnf_id
        self.type = vnf_type
        self.cpu_required = 0
        self.memory_required = 0
        self.storage_required = 0
        self.bandwidth_required = 0
        self.max_latency = 0
        self.placement_domain = None
        
        self._set_requirements()
    
    def _set_requirements(self):
        requirements = {
            'video-processing': {
                'cpu': 30, 'memory': 60, 'storage': 100,
                'bandwidth': 50, 'latency': 50
            },
            'ai-inference': {
                'cpu': 40, 'memory': 80, 'storage': 150,
                'bandwidth': 30, 'latency': 30
            },
            'database-cache': {
                'cpu': 20, 'memory': 100, 'storage': 200,
                'bandwidth': 40, 'latency': 20
            },
            'cdn-edge': {
                'cpu': 25, 'memory': 50, 'storage': 300,
                'bandwidth': 100, 'latency': 15
            },
            'iot-gateway': {
                'cpu': 15, 'memory': 30, 'storage': 50,
                'bandwidth': 20, 'latency': 40
            }
        }
        
        req = requirements.get(self.type, requirements['video-processing'])
        self.cpu_required = req['cpu']
        self.memory_required = req['memory']
        self.storage_required = req['storage']
        self.bandwidth_required = req['bandwidth']
        self.max_latency = req['latency']
    
    def __repr__(self):
        return f"VNF({self.id}, Type: {self.type})"


class Infrastructure:
    
    def __init__(self):
        self.domains = {}
        self.links = []
        self.network_graph = nx.Graph()
        
    def add_domain(self, domain: CloudDomain):
        self.domains[domain.id] = domain
        self.network_graph.add_node(domain.id, domain=domain)
    
    def add_link(self, link: NetworkLink):
        self.links.append(link)
        self.network_graph.add_edge(
            link.source, link.destination,
            bandwidth=link.bandwidth_capacity,
            latency=link.latency,
            link=link
        )
    
    def get_path_latency(self, source: str, destination: str) -> int:
        try:
            path = nx.shortest_path(self.network_graph, source, destination)
            total_latency = 0
            for i in range(len(path) - 1):
                edge_data = self.network_graph[path[i]][path[i+1]]
                total_latency += edge_data['latency']
            return total_latency
        except:
            return float('inf')
    
    def visualize_topology(self):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.network_graph, k=2, iterations=50)
        
        edge_nodes = [n for n, d in self.network_graph.nodes(data=True) 
                      if d['domain'].type == 'edge']
        regional_nodes = [n for n, d in self.network_graph.nodes(data=True) 
                          if d['domain'].type == 'regional']
        
        nx.draw_networkx_nodes(self.network_graph, pos, nodelist=edge_nodes,
                              node_color='lightblue', node_size=2000, 
                              label='Edge Cloud')
        nx.draw_networkx_nodes(self.network_graph, pos, nodelist=regional_nodes,
                              node_color='lightcoral', node_size=3000,
                              label='Regional Cloud')
        
        nx.draw_networkx_edges(self.network_graph, pos, width=2, alpha=0.6)
        
        labels = {n: d['domain'].name for n, d in self.network_graph.nodes(data=True)}
        nx.draw_networkx_labels(self.network_graph, pos, labels, font_size=10)
        
        plt.title("Cloud-Network Infrastructure Topology", fontsize=16)
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/plots/infrastructure_topology.png', dpi=300, bbox_inches='tight')
        print("    ✓ Topology saved as 'results/plots/infrastructure_topology.png'")



class ILPVNFPlacement:
    
    def __init__(self, infrastructure: Infrastructure):
        self.infrastructure = infrastructure
        
    def place_vnfs(self, vnfs: List[VNF]) -> Tuple[Dict, float]:
        # Create the optimization problem
        prob = LpProblem("VNF_Placement", LpMinimize)
        
        domains = list(self.infrastructure.domains.keys())
        vnf_ids = [vnf.id for vnf in vnfs]
        
        x = LpVariable.dicts("placement", 
                            ((v, d) for v in vnf_ids for d in domains),
                            cat='Binary')
        
        objective_terms = []
        
        for vnf in vnfs:
            for domain_id in domains:
                domain = self.infrastructure.domains[domain_id]
                resource_cost = vnf.cpu_required + vnf.memory_required * 0.5
                latency_penalty = 10 if domain.type == 'regional' else 0
                cost = resource_cost + latency_penalty
                objective_terms.append(cost * x[(vnf.id, domain_id)])
        
        prob += lpSum(objective_terms), "Total_Cost"
        
        for vnf in vnfs:
            prob += lpSum([x[(vnf.id, d)] for d in domains]) == 1, \
                   f"Place_VNF_{vnf.id}"
        
        for domain_id in domains:
            domain = self.infrastructure.domains[domain_id]
            
            # CPU capacity
            prob += lpSum([vnf.cpu_required * x[(vnf.id, domain_id)] 
                          for vnf in vnfs]) <= domain.cpu_capacity, \
                   f"CPU_Capacity_{domain_id}"
            
            # Memory capacity
            prob += lpSum([vnf.memory_required * x[(vnf.id, domain_id)] 
                          for vnf in vnfs]) <= domain.memory_capacity, \
                   f"Memory_Capacity_{domain_id}"
        
        prob.solve(PULP_CBC_CMD(msg=0))
        
      
        placement = {}
        for vnf in vnfs:
            for domain_id in domains:
                if value(x[(vnf.id, domain_id)]) == 1:
                    placement[vnf.id] = domain_id
                    break
        
        return placement, value(prob.objective)
        
        return placement, value(prob.objective)

class GreedyPlacement:
    """Greedy algorithm for VNF placement"""
    
    def __init__(self, infrastructure: Infrastructure):
        self.infrastructure = infrastructure
        
    def place_vnfs(self, vnfs: List[VNF]) -> Dict:
        placement = {}
        
        for vnf in vnfs:
            best_domain = None
            best_score = -1
            
            for domain_id, domain in self.infrastructure.domains.items():
                if domain.can_host(vnf):
                    avail_cpu = domain.cpu_capacity - domain.cpu_used
                    avail_memory = domain.memory_capacity - domain.memory_used
                    
                    type_bonus = 50 if domain.type == 'edge' else 0
                    score = avail_cpu + avail_memory * 0.5 + type_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_domain = domain
            
            if best_domain:
                best_domain.allocate_vnf(vnf)
                placement[vnf.id] = best_domain.id
                vnf.placement_domain = best_domain.id
        
        return placement


class FirstFitPlacement:
    
    def __init__(self, infrastructure: Infrastructure):
        self.infrastructure = infrastructure
        
    def place_vnfs(self, vnfs: List[VNF]) -> Dict:
        placement = {}
        
        for vnf in vnfs:
            for domain_id, domain in self.infrastructure.domains.items():
                if domain.can_host(vnf):
                    domain.allocate_vnf(vnf)
                    placement[vnf.id] = domain_id
                    vnf.placement_domain = domain_id
                    break
        
        return placement


class BestFitPlacement:
    
    def __init__(self, infrastructure: Infrastructure):
        self.infrastructure = infrastructure
        
    def place_vnfs(self, vnfs: List[VNF]) -> Dict:
        placement = {}
        
        for vnf in vnfs:
            best_domain = None
            min_waste = float('inf')
            
            for domain_id, domain in self.infrastructure.domains.items():
                if domain.can_host(vnf):
                    avail_cpu = domain.cpu_capacity - domain.cpu_used
                    avail_memory = domain.memory_capacity - domain.memory_used
                    waste = (avail_cpu - vnf.cpu_required) + \
                           (avail_memory - vnf.memory_required)
                    
                    if waste < min_waste:
                        min_waste = waste
                        best_domain = domain
            
            if best_domain:
                best_domain.allocate_vnf(vnf)
                placement[vnf.id] = best_domain.id
                vnf.placement_domain = best_domain.id
        
        return placement


class PerformanceEvaluator:
    
    def __init__(self, infrastructure: Infrastructure):
        self.infrastructure = infrastructure
        
    def calculate_metrics(self, vnfs: List[VNF], 
                         placement: Dict) -> Dict:
        metrics = {
            'acceptance_ratio': 0.0,
            'average_latency': 0.0,
            'cpu_utilization': 0.0,
            'memory_utilization': 0.0,
            'resource_efficiency': 0.0,
            'placed_vnfs': 0,
            'rejected_vnfs': 0
        }
        
        placed = len(placement)
        total = len(vnfs)
        metrics['acceptance_ratio'] = (placed / total * 100) if total > 0 else 0
        metrics['placed_vnfs'] = placed
        metrics['rejected_vnfs'] = total - placed
        
        latencies = []
        for vnf_id, domain_id in placement.items():
            domain = self.infrastructure.domains[domain_id]
            latency = 15 if domain.type == 'edge' else 35
            latencies.append(latency)
        
        metrics['average_latency'] = np.mean(latencies) if latencies else 0
        total_cpu = sum(d.cpu_capacity for d in self.infrastructure.domains.values())
        used_cpu = sum(d.cpu_used for d in self.infrastructure.domains.values())
        metrics['cpu_utilization'] = (used_cpu / total_cpu * 100) if total_cpu > 0 else 0
        
        total_memory = sum(d.memory_capacity for d in self.infrastructure.domains.values())
        used_memory = sum(d.memory_used for d in self.infrastructure.domains.values())
        metrics['memory_utilization'] = (used_memory / total_memory * 100) if total_memory > 0 else 0
        
        metrics['resource_efficiency'] = placed / (used_cpu + used_memory) if (used_cpu + used_memory) > 0 else 0
        
        return metrics
    
    def compare_algorithms(self, vnfs: List[VNF]) -> pd.DataFrame:
        results = []
        
        algorithms = {
            'Greedy': GreedyPlacement(self.infrastructure),
            'First-Fit': FirstFitPlacement(self.infrastructure),
            'Best-Fit': BestFitPlacement(self.infrastructure)
        }
        
        for alg_name, algorithm in algorithms.items():
            for domain in self.infrastructure.domains.values():
                domain.cpu_used = 0
                domain.memory_used = 0
                domain.storage_used = 0
                domain.hosted_vnfs = []
            
            placement = algorithm.place_vnfs(vnfs)
            metrics = self.calculate_metrics(vnfs, placement)
            metrics['algorithm'] = alg_name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def visualize_comparison(self, comparison_df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('VNF Placement Algorithm Comparison', fontsize=16)
        
        metrics = ['acceptance_ratio', 'average_latency', 
                  'cpu_utilization', 'memory_utilization']
        titles = ['Acceptance Ratio (%)', 'Average Latency (ms)',
                 'CPU Utilization (%)', 'Memory Utilization (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            comparison_df.plot(x='algorithm', y=metric, kind='bar', 
                             ax=ax, legend=False, color='skyblue')
            ax.set_title(title)
            ax.set_xlabel('Algorithm')
            ax.set_ylabel(title.split('(')[0].strip())
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/algorithm_comparison.png', dpi=300, bbox_inches='tight')
        print("    ✓ Comparison saved as 'results/plots/algorithm_comparison.png'")


def create_sample_infrastructure() -> Infrastructure:
    infra = Infrastructure()
    
    infra.add_domain(CloudDomain('edge1', 'Edge Cloud 1', 'edge', 
                                 cpu=100, memory=200, storage=500))
    infra.add_domain(CloudDomain('edge2', 'Edge Cloud 2', 'edge',
                                 cpu=100, memory=200, storage=500))
    infra.add_domain(CloudDomain('edge3', 'Edge Cloud 3', 'edge',
                                 cpu=100, memory=200, storage=500))
    infra.add_domain(CloudDomain('regional1', 'Regional Cloud 1', 'regional',
                                 cpu=300, memory=600, storage=2000))
    infra.add_domain(CloudDomain('regional2', 'Regional Cloud 2', 'regional',
                                 cpu=300, memory=600, storage=2000))
    
    # Add network links
    # Edge to Regional connections
    infra.add_link(NetworkLink('edge1', 'regional1', bandwidth=1000, latency=10))
    infra.add_link(NetworkLink('edge2', 'regional1', bandwidth=1000, latency=10))
    infra.add_link(NetworkLink('edge2', 'regional2', bandwidth=1000, latency=10))
    infra.add_link(NetworkLink('edge3', 'regional2', bandwidth=1000, latency=10))
    infra.add_link(NetworkLink('regional1', 'regional2', bandwidth=2000, latency=15))
    
    return infra


def generate_vnf_workload(num_vnfs: int) -> List[VNF]:
    vnf_types = ['video-processing', 'ai-inference', 'database-cache', 
                 'cdn-edge', 'iot-gateway']
    vnfs = []
    
    for i in range(num_vnfs):
        vnf_type = random.choice(vnf_types)
        vnf = VNF(f'vnf_{i}', vnf_type)
        vnfs.append(vnf)
    
    return vnfs


def run_simulation():
    print("=" * 70)
    print("CFN VNF PLACEMENT SIMULATOR FOR BEYOND 5G")
    print("=" * 70)
    
    print("\n[1/5] Creating multi-domain infrastructure...")
    infra = create_sample_infrastructure()
    print(f"    ✓ Created {len(infra.domains)} cloud domains")
    print(f"    ✓ Created {len(infra.links)} network links")
    
    print("\n[2/5] Visualizing network topology...")
    infra.visualize_topology()
    
    print("\n[3/5] Generating VNF workload...")
    num_vnfs = 20
    vnfs = generate_vnf_workload(num_vnfs)
    print(f"    ✓ Generated {num_vnfs} VNF requests")
    
    vnf_type_counts = {}
    for vnf in vnfs:
        vnf_type_counts[vnf.type] = vnf_type_counts.get(vnf.type, 0) + 1
    
    print("\n    VNF Distribution:")
    for vnf_type, count in vnf_type_counts.items():
        print(f"      - {vnf_type}: {count}")
    
    print("\n[4/5] Running placement algorithms...")
    evaluator = PerformanceEvaluator(infra)
    comparison_results = evaluator.compare_algorithms(vnfs)
    
    print("\n    Results Summary:")
    print(comparison_results.to_string(index=False))
    
    print("\n[5/5] Generating visualizations...")
    evaluator.visualize_comparison(comparison_results)
    
    # Save detailed results
    comparison_results.to_csv('results/metrics/placement_results.csv', index=False)
    print("    ✓ Results saved to 'results/metrics/placement_results.csv'")
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. results/plots/infrastructure_topology.png")
    print("  2. results/plots/algorithm_comparison.png")
    print("  3. results/metrics/placement_results.csv")
    print("\n")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    run_simulation()
    
    print("\nNext Steps:")
    print("  • Check the results folder for outputs")
    print("  • Modify parameters and re-run")
    print("  • Implement your own algorithms")