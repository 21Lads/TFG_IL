# TFG — Driving Simulation using Maximum Entropy Inverse Reinforcement Learning

Driving simulation and behavior modeling project based on **Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)** and imitation learning techniques.

Dataset used in this project:  
https://www.it.uc3m.es/madrid-traces/



## Overview

This project implements a driving simulation pipeline using:

- Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)
- Imitation learning
- Highway driving simulation
- Driving style comparison experiments

The main workflow is demonstrated in `main.ipynb`, which guides through the complete pipeline from data processing to policy evaluation.



## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install dependencies

Make sure the required Python packages are installed.

Important dependency:

```bash
pip install highway-env
```

You may also need:

```bash
pip install numpy pandas matplotlib gymnasium pygame
```



## Usage

## Running the Full Pipeline

The notebook below walks through the entire project pipeline:

```bash
main.ipynb
```

> Note: The highway simulation inside the notebook only displays aggregate statistics.

To visualize the simulation interactively, run `Simulator.py` separately after saving the required files into the `/results` directory (unless they already exist).



## Running the Highway Simulator

### Steps

1. Ensure all dependencies are installed, especially `highway-env`
2. Make sure the required result files are available in `/results`
3. Run:

```bash
python Simulator.py
```

A new simulation window will open displaying:

- A blue vehicle
- A three-lane highway environment



## Simulator Controls

| Key | Action |
|||
| `SPACE` | Pause / Resume simulation |
| `n` | Advance one timestep |
| `p` | Capture probe/frame for inspection |
| `q` | Quit simulation |



## Driving Style Experiments

### Visualizing Experiment 2

Experiment 2 compares contrasting driving styles and can be visualized directly inside:

```python
experiment2 = True
```

in `main.ipynb`.

### Re-running the Experiment

To execute the experiment again:

```bash
python mainDrivingStylesExperiment.py
```



## Methodology

This project explores:

- Learning reward functions from expert driving trajectories
- Modeling driver behavior using maximum entropy inverse reinforcement learning
- Visualising trajectories and kinematic profiles
- Simulating learned policies in a highway environment
- Comparing distinct driving styles



## License

MIT License

```text
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```