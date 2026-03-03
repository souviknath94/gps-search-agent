# GPS Search Agent: Finding the Optimal Path

An implementation of **Local Search (Genetic Algorithm)** to solve a pathfinding problem.

## 📍 Problem Statement
Vyuha, a software engineer, needs to travel from her native place, **Panaji**, back to her office in **Chennai**. Due to technical issues with standard mapping tools and the unavailability of flights, she must rely on road transport. 

The goal of this agent is to find the most optimal road path covering various cities in South India while minimizing the total distance travelled.

![Road Connectivity Map](images/map.png)  <-- THIS IS THE FINAL LINE

---

## 🏗️ Project Components

The project is structured modularly to separate data handling, the genetic engine, and the search models:

- **`src/utils.py`**: Contains the `haversine` formula for great-circle distance calculations and `DataUtils` for cleaning geographical data.
- **`src/models.py`**: Defines the `Chromosome` (path representation) and the `FitnessFunction` (rewarding valid, short paths to the goal).
- **`src/ga_engine.py`**: The core Genetic Algorithm implementation, including Roulette Wheel selection, Edge Recombination crossover, and Scramble mutation.
- **`main.py`**: The entry point to run the simulation and find the best path.

---

## 🚀 Getting Started

This project uses [**uv**](https://docs.astral.sh/uv/) for extremely fast Python package and project management.

### 1. Prerequisites
You need to have `uv` installed on your system. 
> **Note:** If you don't have it yet, follow the installation guide on the [Official uv Website](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Environment Setup
Clone the repository and run the following command to create a virtual environment and install all dependencies (Pandas, Numpy, etc.) automatically:

```bash
# Sync the environment and install dependencies
uv sync