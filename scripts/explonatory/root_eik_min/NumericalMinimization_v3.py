#  Generic ROOT Minimization with Confidence Level Calculation
#  Supports any function with Hesse and MINOS error analysis

import ROOT
import numpy as np
import math


def GenericMinimization(func,
                        ndim,
                        minimizerName="Minuit2",
                        algoName="",
                        startPoint=None,
                        stepSize=None,
                        randomSeed=-1,
                        confidenceLevel=90.0):
    """
    Generic ROOT Minimization Wrapper with Confidence Level Calculation

    Parameters
    ----------
    func : callable
        Function to minimize. Should take a numpy array or list of length ndim.
    ndim : int
        Number of dimensions (number of parameters).
    minimizerName : str, default="Minuit2"
        Minimizer name (Minuit, Minuit2, GSLMultiMin, GSLSimAn, Genetic).
    algoName : str, default=""
        Specific algorithm (Migrad, BFGS, ConjugateFR, Simplex, etc.).
    startPoint : list or None
        Initial guess for variables. If None, defaults to zeros.
    stepSize : list or None
        Step sizes for each variable. If None, defaults to 0.01 for all.
    randomSeed : int, default=-1
        Random initialization of start point if >= 0 (uniform in [-5, 5]).
    maxFunctionCalls : int, default=1000000
        Maximum allowed function evaluations.
    maxIterations : int, default=10000
        Maximum allowed iterations.
    tolerance : float, default=0.001
        Desired tolerance for convergence.
    printLevel : int, default=1
        Verbosity level of minimizer (0=quiet, 1=normal, 2=verbose).
    confidenceLevel : float, default=90.0
        Confidence level for error calculation (68.3, 90.0, 95.0, or 99.0).

    Returns
    -------
    dict
        Dictionary containing:
        - 'success': bool, whether minimization succeeded
        - 'x': numpy array, parameter values at minimum
        - 'fval': float, function value at minimum
        - 'status': int, minimizer status
        - 'ncalls': int, number of function calls
        - 'edm': float, estimated distance to minimum
        - 'hesse_errors': numpy array, symmetric errors from Hesse
        - 'minos_errors_low': numpy array, lower MINOS errors
        - 'minos_errors_up': numpy array, upper MINOS errors
        - 'correlation': numpy array, correlation matrix
    """
    
    # Set default values
    if startPoint is None:
        startPoint = [0.0] * ndim
    if stepSize is None:
        stepSize = [0.01] * ndim
    
    # Map confidence level to ErrorDef for 4D case
    # Based on chi-square distribution with 4 degrees of freedom
    cl_to_errordef_4d = {
        68.3: 4.72,
        90.0: 7.78,
        95.0: 9.49,
        99.0: 13.28
    }
    
    # For 4 parameters (from chi-square distribution)
    if confidenceLevel in cl_to_errordef_4d:
        errordef = cl_to_errordef_4d[confidenceLevel]
    else:
        # Default to 1.0 if CL not recognized
        print(f"Warning: CL {confidenceLevel}% not standard. Using ErrorDef=1.0")
        errordef = 1.0
    
    print(f"\n{'='*70}")
    print(f"Generic Minimization - {ndim}D Function")
    print(f"Minimizer: {minimizerName} {algoName}")
    print(f"Confidence Level: {confidenceLevel}%")
    print(f"{'='*70}\n")

    # Create minimizer
    minimizer = ROOT.Math.Factory.CreateMinimizer(minimizerName, algoName)
    if not minimizer:
        raise RuntimeError(f"Cannot create minimizer \"{minimizerName}\"")

    # Configure minimizer
    minimizer.SetMaxFunctionCalls(10000000)
    minimizer.SetMaxIterations(10000)
    minimizer.SetTolerance(1e-8)
    minimizer.SetPrintLevel(1)
    minimizer.SetErrorDef(errordef)

    # Create function wrapper
    f = ROOT.Math.Functor(func, ndim)
    minimizer.SetFunction(f)
    
    # Random starting point if requested
    variable = list(startPoint)
    if randomSeed >= 0:
        r = ROOT.TRandom2(randomSeed)
        variable = [r.Uniform(-5, 5) for _ in range(ndim)]
        print(f"Using random start point (seed={randomSeed})")
    
    # Set variables
    for i in range(ndim):
        minimizer.SetVariable(i, f"x{i}", variable[i], stepSize[i])
    
    print(f"Starting point: {variable}")
    print(f"Beginning minimization...\n")

    # Run minimization
    ret = minimizer.Minimize()

    if not ret:
        print("ERROR: Minimization failed!")
        return {'success': False}

    # Get results
    xs = np.array([minimizer.X()[i] for i in range(ndim)])
    hesse_errors = np.array([minimizer.Errors()[i] for i in range(ndim)])
    
    print(f"\n{'='*70}")
    print(f"MINIMIZATION RESULTS:")
    print(f"{'='*70}")
    print(f"Status: {minimizer.Status()} (0 = success)")
    print(f"Function value at minimum: f = {minimizer.MinValue():.10e}")
    print(f"Number of function calls: {minimizer.NCalls()}")
    print(f"EDM (Estimated Distance to Minimum): {minimizer.Edm():.10e}")
    print(f"\nParameter values:")
    for i in range(ndim):
        print(f"  x{i} = {xs[i]:.10f}")
    
    # Hesse errors at specified confidence level
    print(f"\n{'='*70}")
    print(f"HESSE ERRORS (Symmetric, {confidenceLevel}% CL):")
    print(f"{'='*70}")
    for i in range(ndim):
        print(f"  x{i} = {xs[i]:.8f} ± {hesse_errors[i]:.8f}")
    
    # MINOS errors at specified confidence level
    print(f"\n{'='*70}")
    print(f"MINOS ERRORS (Asymmetric, {confidenceLevel}% CL):")
    print(f"{'='*70}")
    
    minos_errors_low = np.zeros(ndim)
    minos_errors_up = np.zeros(ndim)
    
    for i in range(ndim):
        errLow = np.array([0.0], dtype=np.float64)
        errUp = np.array([0.0], dtype=np.float64)
        success = minimizer.GetMinosError(i, errLow, errUp)
        
        if success:
            minos_errors_low[i] = errLow[0]
            minos_errors_up[i] = errUp[0]
            print(f"  x{i} = {xs[i]:.8f} {errLow[0]:+.8f} / {errUp[0]:+.8f}")
        else:
            print(f"  x{i}: MINOS failed")
            minos_errors_low[i] = -hesse_errors[i]
            minos_errors_up[i] = hesse_errors[i]
    
    # Correlation matrix
    print(f"\n{'='*70}")
    print(f"CORRELATION MATRIX:")
    print(f"{'='*70}")
    
    correlation = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            cov_ij = minimizer.CovMatrix(i, j)
            cov_ii = minimizer.CovMatrix(i, i)
            cov_jj = minimizer.CovMatrix(j, j)
            if cov_ii > 0 and cov_jj > 0:
                correlation[i, j] = cov_ij / math.sqrt(cov_ii * cov_jj)
    
    # Print correlation matrix
    header = "     " + "".join([f"  x{i:2d}    " for i in range(ndim)])
    print(header)
    for i in range(ndim):
        row = f"x{i:2d}  "
        for j in range(ndim):
            row += f"{correlation[i, j]:7.4f}  "
        print(row)
    
    print(f"\n{'='*70}")
    if ret and minimizer.Status() == 0:
        print("✓ Minimization converged successfully!")
    else:
        print("✗ Minimization did not fully converge")
    print(f"{'='*70}\n")
    
    # Return results dictionary
    return {
        'success': ret and minimizer.Status() == 0,
        'x': xs,
        'fval': minimizer.MinValue(),
        'status': minimizer.Status(),
        'ncalls': minimizer.NCalls(),
        'edm': minimizer.Edm(),
        'hesse_errors': hesse_errors,
        'minos_errors_low': minos_errors_low,
        'minos_errors_up': minos_errors_up,
        'correlation': correlation
    }


# 4D Test Function
def StyblinskiTang4D(vecx):
    """
    Styblinski-Tang function in 4D
    Global minimum at x_i = -2.903534 for all i
    f(x*) = -156.66468... (for 4D)
    """
    result = 0.0
    for i in range(4):
        x = vecx[i]
        result += x**4 - 16*x**2 + 5*x
    return result / 2.0


if __name__ == "__main__":
    # 4D Styblinski-Tang Function at 90% CL
    print("\n" + "="*70)
    print("4D Styblinski-Tang Function Minimization")
    print("="*70)
    result = GenericMinimization(
        func=StyblinskiTang4D,
        ndim=4,
        startPoint=[-1.0, -1.0, -1.0, -1.0],
        confidenceLevel=90.0
    )