# Computing First Networking VNF Placement Simulator

> A comprehensive simulation framework for Virtual Network Function (VNF) placement in multi-domain Beyond 5G cloud-network infrastructures.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![React Version](https://img.shields.io/badge/react-18.0%2B-61dafb)](https://reactjs.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![Demo](docs/demo-screenshot.png)

## 🎯 Project Overview

This project implements and compares multiple VNF placement algorithms for Beyond 5G networks, addressing the critical challenge of resource orchestration in cloud-network integrated infrastructures.

### Key Features

- **Multi-Algorithm Implementation**: Greedy, First-Fit, Best-Fit, and ILP-based placement
- **Real-time Visualization**: Interactive web interface with live metrics
- **Performance Evaluation**: Comprehensive metrics including acceptance ratio, latency, and resource utilization
- **Extensible Architecture**: Easy to add new algorithms and VNF types

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                Cloud Infrastructure                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Edge     │  │ Edge     │  │ Edge     │         │
│  │ Cloud 1  │  │ Cloud 2  │  │ Cloud 3  │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│       │             │             │                 │
│       └─────────────┼─────────────┘                │
│                     │                               │
│       ┌─────────────┴─────────────┐                │
│       │                           │                 │
│  ┌────▼─────┐              ┌─────▼────┐           │
│  │Regional  │◄────────────►│Regional  │           │
│  │Cloud 1   │              │Cloud 2   │           │
│  └──────────┘              └──────────┘           │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher (for web interface)
- pip and npm

### Installation

1. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Python simulator**
   ```bash
   python main.py
   ```

3. **Set up web interface (optional)**
   ```bash
   cd web-simulator/cfn-web-app
   npm install
   npm start
   ```

## 📊 Usage


**Custom parameters:**
```python
# Edit main.py
num_vnfs = 50  


python scripts/run_experiments.py
```

### Viewing Results

Results are saved in `results/` directory:
- `results/plots/` - PNG visualizations
- `results/metrics/` - CSV data files
- `results/logs/` - Simulation logs

## 🧪 Algorithms Implemented

### 1. Greedy Placement
Selects the domain with the highest score based on available resources and latency preferences.

**Time Complexity:** O(n × m) where n = VNFs, m = domains

### 2. First-Fit Placement
Places VNF in the first domain with sufficient resources.

**Time Complexity:** O(n × m)

### 3. Best-Fit Placement
Selects the domain that minimizes resource waste.

**Time Complexity:** O(n × m)

### 4. ILP-Based Placement
Optimal placement using Integer Linear Programming.

**Time Complexity:** NP-hard (exponential worst case)


### Adding New Algorithms

Create a new class in `main.py`:

```python
class MyCustomPlacement:
    def __init__(self, infrastructure):
        self.infrastructure = infrastructure
    
    def place_vnfs(self, vnfs):
        # Your algorithm logic here
        placement = {}
        # ... implementation
        return placement
```

### Adding New VNF Types

Add to `VNF._set_requirements()`:

```python
'new-vnf-type': {
    'cpu': 35, 
    'memory': 70, 
    'storage': 120,
    'bandwidth': 45, 
    'latency': 25
}
```





### Algorithm Comparison

| Algorithm | Acceptance (%) | Latency (ms) | CPU Util (%) |
|-----------|----------------|--------------|--------------|
| Greedy    | 95.0          | 22.5         | 68.2         |
| First-Fit | 85.0          | 28.3         | 64.1         |
| Best-Fit  | 90.0          | 25.1         | 71.5         |

*Results based on 20 VNF workload across 5 cloud domains*

