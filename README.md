# Atomic Cluster Optimization and Analysis ⚛️✨

This repository contains tools for the optimization, analysis, and simulation of atomic structures using the Gupta potential. 🧪 It is designed to work with XYZ files, which represent atomic configurations in three-dimensional space. 🌌

## Main Features 🚀

1. **Energy Calculation**: ⚡ Compute the total energy of an atomic structure using the Gupta potential.
2. **Force Calculation**: 🌀 Determine the gradient (forces) of the energy with respect to atomic coordinates.
3. **Hessian Calculation**: 📈 Compute the Hessian matrix, which describes the second derivatives of the energy.
4. **Structure Optimization**: 🔧 Optimize the atomic configuration to minimize the total energy.
5. **Random Configuration Generation**: 🎲 Create random initial configurations of atoms.
6. **Bond Calculation**: 🔗 Determine interatomic distances and valid bonds between atoms.

## Gupta Potential 🧪✨

The file [`potentials/gupta.py`](potentials/gupta.py) implements the Gupta potential, an empirical model used to describe transition metal systems. 🛠️ This potential includes interactions between iron (Fe), cobalt (Co), and nickel (Ni) atoms. 🧲

### Features of the Gupta Potential 🌟:
- **Cohesive Energy**: 💎 Models the bonding energy between atoms.
- **Elastic Constants**: 🧬 Describes the mechanical properties of the system.
- **Adjustable Parameters**: ⚙️ Specific parameters for each atom pair (Fe-Fe, Fe-Co, etc.) are defined in the file.

### Available Methods 📚:
- `potential(coords)`: 🧮 Calculates the potential energy of an atomic configuration.
- `gradient(coords)`: 🌀 Computes the energy gradient (forces).
- `hessian(coords)`: 📊 Computes the Hessian matrix.

## Requirements 📦

To use the tools in this repository, ensure you have the following Python packages installed:

- `numpy` 🧮: For numerical computations.
- `scipy` 📊: For optimization and scientific calculations.
- `autograd` 🔧: For automatic differentiation and gradient computations.
