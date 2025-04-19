# Script to optimize atomic structures from an XYZ file using the Gupta potential
import argparse
import numpy as np
import scipy.optimize as spo
from autograd import elementwise_grad as egrad
from autograd import hessian as hess
from potentials.gupta import Gupta
from read_xyz import read_xyz_file
from write_xyz import write_xyz_file
from timer import timer

# @timer
def optimize_structure(file_path, method="L-BFGS-B", tol=1e-8, maxiter=1000):
    """Optimize the atomic structure from an XYZ file using the Gupta potential.
    
    Args:
        file_path (str): Path to the input XYZ file.
        method (str): Optimization method. Default is "L-BFGS-B".
        tol (float): Convergence tolerance. Default is 1e-8.
        maxiter (int): Maximum number of iterations. Default is 1000.
    
    Returns:
        None: Saves the optimized structure to a new XYZ file.
    """
    atoms, coords = read_xyz_file(file_path)
    
    gupta = Gupta(atoms)
    potential = lambda x: gupta.potential(x.reshape(len(coords), 3))
    
    # Define which methods need gradient and hessian
    gradient_methods = ["CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP"]
    hessian_methods = ["Newton-CG", "trust-ncg", "trust-krylov", "trust-exact", "dogleg", "trust-constr"]
    no_derivative_methods = ["Nelder-Mead", "Powell"]
    global_methods = ["basinhopping", "differential_evolution"]
    
    # Prepare optimization options
    options = {
        "maxiter": maxiter,
        "disp": False
    }
    
    # Add method-specific options and tolerances
    if method in gradient_methods:
        options["gtol"] = tol
    elif method in ["trust-constr"]:
        options["gtol"] = tol
        options["xtol"] = tol
    elif method in no_derivative_methods:
        options["xatol"] = tol
        options["fatol"] = tol
    
    # Handle global optimization methods
    if method == 'basinhopping':
        sol = spo.basinhopping(
                potential,  # The potential energy function to minimize
                coords.flatten(),  # Initial guess for the atomic coordinates
                minimizer_kwargs={
                    "method": "L-BFGS-B",  # Local minimization method
                    "jac": egrad(potential),  # Gradient of the potential energy
                },
                niter=250,  # Number of basin-hopping iterations
                disp=False)  # Suppress output display
    elif method == 'differential_evolution':
        # Determine reasonable bounds for the coordinates
        # Use the initial structure +/- a margin
        margin = 5.0  # Margin of 5 Å for exploration
        flattened_coords = coords.flatten()
        bounds = [(x - margin, x + margin) for x in flattened_coords]
        
        # Configure and execute differential_evolution
        sol = spo.differential_evolution(
            func=potential,
            bounds=bounds,
            strategy='best1bin',
            maxiter=maxiter,
            popsize=15,
            tol=tol,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=None,
            callback=None,
            disp=False,
            polish=True,  # Apply local optimization at the end
            init='latinhypercube',
            atol=tol,
            updating='immediate',
            workers=1
        )
    else:
        # Prepare arguments for minimize
        kwargs = {
            "fun": potential,
            "x0": coords.flatten(),
            "method": method,
            "options": options
        }
        
        # Add gradient if needed
        if method in gradient_methods + hessian_methods:
            kwargs["jac"] = egrad(potential)
        
        # Add hessian if needed
        if method in hessian_methods:
            # Adapt hessian calculation based on the method
            if method == "Newton-CG":
                # Newton-CG accepts a callable for Hessian-vector product
                hess_p = lambda x, p: hess(potential)(x).dot(p)
                kwargs["hessp"] = hess_p
            else:
                # Other methods that use explicit hessian
                kwargs["hess"] = hess(potential)
        
        sol = spo.minimize(**kwargs)
    
    new_coords = sol.x.reshape(-1, 3)
    energy = sol.fun
    
    output_file = f"opt-{file_path.split('/')[-1]}"
    write_xyz_file(output_file, atoms, new_coords, str(energy))
    
    print(f"Old energy: {potential(coords.flatten())} eV | New energy: {energy} eV")
    print(f"Optimization complete. Output saved to {output_file}")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Optimize atomic structure from an XYZ file.")
        parser.add_argument("file", type=str, help="Path to the XYZ file.")
        parser.add_argument("--method", type=str, default="L-BFGS-B", 
                            choices=["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP", 
                                    "Newton-CG", "trust-ncg", "trust-krylov", "trust-exact", 
                                    "dogleg", "trust-constr", "basinhopping", "differential_evolution"],
                            help="Optimization method to use. Default is 'L-BFGS-B'.")
        parser.add_argument("--tol", type=float, default=1e-8,
                            help="Convergence tolerance. Default is 1e-8.")
        parser.add_argument("--maxiter", type=int, default=1000,
                            help="Maximum number of iterations. Default is 1000.")
        args = parser.parse_args()
        
        if not args.file:
            raise ValueError("No file path provided. Please specify the path to an XYZ file.")
        
        optimize_structure(args.file, method=args.method, tol=args.tol, maxiter=args.maxiter)
    except Exception as e:
        print(f"Error: {e}")
