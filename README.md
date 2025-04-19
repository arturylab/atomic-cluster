# Atomic Cluster Optimization and Analysis ⚛️✨

This repository contains tools for the optimization, analysis, and simulation of atomic structures using the Gupta potential. 🧪 It is designed to work with XYZ files, which represent atomic configurations in three-dimensional space. 🌌

## Main Features 🚀

1. **Energy Calculation**: ⚡ Compute the total energy of an atomic structure using the Gupta potential.
2. **Force Calculation**: 🌀 Determine the gradient (forces) of the energy with respect to atomic coordinates.
3. **Hessian Calculation**: 📈 Compute the Hessian matrix, which describes the second derivatives of the energy.
4. **Structure Optimization**: 🔧 Optimize the atomic configuration to minimize the total energy with multiple methods.
5. **Random Configuration Generation**: 🎲 Create random initial configurations of atoms with minimum distance constraints.

## Gupta Potential 🧪✨

The file [`potentials/gupta.py`](potentials/gupta.py) implements the Gupta potential, an empirical model used to describe transition metal systems. 🛠️ This potential includes interactions between various transition metals including iron (Fe), cobalt (Co), nickel (Ni), copper (Cu), palladium (Pd), silver (Ag), platinum (Pt), and gold (Au). 🧲

### Features of the Gupta Potential 🌟:
- **Cohesive Energy**: 💎 Models the bonding energy between atoms.
- **Elastic Constants**: 🧬 Describes the mechanical properties of the system.
- **Adjustable Parameters**: ⚙️ Specific parameters for each atom pair are defined in the file.

### Available Methods 📚:
- `potential(coords)`: 🧮 Calculates the potential energy of an atomic configuration.
- `gradient(coords)`: 🌀 Computes the energy gradient (forces).
- `hessian(coords)`: 📊 Computes the Hessian matrix.

### Gupta Potential Parameters 📊

The potential uses five parameters for each atomic pair interaction:
- **A**: Determines the strength of the repulsive potential (eV)
- **XI**: Related to the cohesive energy (eV)
- **P, Q**: Independent elastic constants
- **R0**: Equilibrium lattice parameter (Å)

| Interaction | A | XI | P | Q | R0 |
|-------------|---|----|----|----|----|
| Fe-Fe | 0.13315 | 1.6179 | 10.5000 | 2.6000 | 2.5530 |
| Fe-Co | 0.11246 | 1.5515 | 11.0380 | 2.4379 | 2.5248 |
| Fe-Ni | 0.07075 | 1.3157 | 13.3599 | 1.7582 | 2.5213 |
| Co-Co | 0.09500 | 1.4880 | 11.6040 | 2.2860 | 2.4970 |
| Co-Ni | 0.05970 | 1.2618 | 14.0447 | 1.6486 | 2.4934 |
| Ni-Ni | 0.03760 | 1.0700 | 16.9990 | 1.1890 | 2.4900 |
| Cu-Cu | 0.08550 | 1.2240 | 10.9600 | 2.2780 | 2.5562 |
| Cu-Pd | 0.13005 | 1.4710 | 10.9135 | 3.0100 | 2.6523 |
| Cu-Ag | 0.09800 | 1.2274 | 10.7000 | 2.8050 | 2.7224 |
| Cu-Pt | 0.16000 | 1.8200 | 10.7860 | 3.1410 | 2.6660 |
| Cu-Au | 0.15390 | 1.5605 | 11.0500 | 3.0475 | 2.5562 |
| Pd-Pd | 0.17460 | 1.7180 | 10.8670 | 3.7420 | 2.7485 |
| Pd-Ag | 0.16100 | 1.5597 | 10.8950 | 3.4920 | 2.8185 |
| Pd-Pt | 0.23000 | 2.2000 | 10.7400 | 3.8700 | 2.7600 |
| Pd-Au | 0.19000 | 1.7500 | 10.5400 | 3.8900 | 2.8160 |
| Ag-Ag | 0.10280 | 1.1780 | 10.9280 | 3.1390 | 2.8885 |
| Ag-Pt | 0.17500 | 1.7900 | 10.7300 | 3.5900 | 2.8330 |
| Ag-Au | 0.14900 | 1.4874 | 10.4940 | 3.6070 | 2.8864 |
| Pt-Pt | 0.29750 | 2.6950 | 10.6120 | 4.0040 | 2.7747 |
| Pt-Au | 0.25000 | 2.2000 | 10.4200 | 4.0200 | 2.8300 |
| Au-Au | 0.20610 | 1.7900 | 10.2290 | 4.0360 | 2.8843 |

## Optimization Methods 🔍

The optimizer supports multiple methods from SciPy's optimization suite:

### Local Optimization Methods:
- **L-BFGS-B** (default): Limited-memory BFGS with bounds
- **BFGS**: Broyden-Fletcher-Goldfarb-Shanno algorithm
- **CG**: Conjugate gradient method
- **Nelder-Mead**: Simplex algorithm (no derivatives required)
- **Powell**: Powell's method (no derivatives required)
- **TNC**: Truncated Newton method
- **SLSQP**: Sequential Least Squares Programming
- **Newton-CG**: Newton's method with conjugate gradient
- **trust-ncg**, **trust-krylov**, **trust-exact**: Trust-region methods
- **dogleg**, **trust-constr**: Additional trust-region methods

### Global Optimization Methods:
- **basinhopping**: Basin-hopping algorithm for finding global minima
- **differential_evolution**: Stochastic method based on evolution of a population

## Requirements 📦

To use the tools in this repository, you'll need the following Python packages:

- `numpy==2.2.4` 🧮: For numerical computations
- `scipy==1.15.2` 📊: For optimization and scientific calculations
- `autograd==1.7.0` 🔧: For automatic differentiation and gradient computations
- `pytest==8.3.5` 🧪: For unit testing

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Usage Examples 📖

### Generate a Random Cluster
```bash
python random_cluster.py Fe5Ni5
```

### Calculate Energy
```bash
python get_energy.py structure.xyz
```

### Calculate Forces
```bash
python get_forces.py structure.xyz --norm
```

### Optimize a Structure
```bash
python optimizer.py structure.xyz --method L-BFGS-B
```

Run with global optimization:
```bash
python optimizer.py structure.xyz --method basinhopping
```
