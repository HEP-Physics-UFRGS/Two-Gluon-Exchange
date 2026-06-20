import ROOT
import numpy as np
import math


def GenericMinimization(func,
                        ndim,
                        minimizerName="Minuit2",
                        algoName="",
                        mg_init=None,
                        eps_init=None,
                        a1_init=None,
                        a2_init=None,
                        stepSize=None,
                        randomSeed=-1,
                        maxFunctionCalls=1000000,
                        maxIterations=10000,
                        tolerance=1e-8,
                        printLevel=1,
                        confidenceLevel=90.0):
    """
    Generic ROOT Minimization Wrapper with Confidence Level Calculation.

    Parameters
    ----------
    func : callable
        Function to minimize. Should accept a list or numpy array of length `ndim`.

    ndim : int
        Number of parameters to minimize.

    minimizerName : str, default="Minuit2"
        Minimizer to use (Minuit, Minuit2, GSLMultiMin, GSLSimAn, Genetic, etc.).
    
    algoName : str, default=""
        Specific algorithm (Migrad, BFGS, ConjugateFR, Simplex, etc.).
    
    mg_init : float or None
        Initial guess for parameter 'mg'. Defaults to 0.0 if None.
    
    eps_init : float or None
        Initial guess for parameter 'eps'. Defaults to 0.0 if None.
    
    a1_init : float or None
        Initial guess for parameter 'a1'. Defaults to 0.0 if None.
    
    a2_init : float or None
        Initial guess for parameter 'a2'. Defaults to 0.0 if None.
    
    stepSize : list of floats or None
        Step sizes for each parameter. Defaults to 0.01 for all.
    
    randomSeed : int, default=-1
        If >=0, use random starting points in [-5, 5].
    
    maxFunctionCalls : int, default=1000000
        Maximum allowed function evaluations.
    
    maxIterations : int, default=10000
        Maximum allowed iterations.
    
    tolerance : float, default=1e-8
        Desired tolerance for convergence.
    
    printLevel : int, default=1
        Verbosity of the minimizer (0=quiet, 1=normal, 2=verbose).
    
    confidenceLevel : float, default=90.0
        Confidence level for error calculation (68.3, 90, 95, 99).

    Returns
    -------
    dict
        Dictionary containing:
        - 'success': bool, whether minimization converged successfully
        - 'x': numpy array, parameter values at minimum
        - 'status': int, minimizer status (0 = success)
        - 'hesse_errors': numpy array, symmetric Hesse errors
        - 'minos_errors_low': numpy array, lower MINOS errors
        - 'minos_errors_up': numpy array, upper MINOS errors
    """

    #-------------------
    #  SET STARTING POINT
    #-------------------

    param_names = ["mg", "eps", "a1", "a2"]
    init_map = [mg_init, eps_init, a1_init, a2_init]
    startPoint = []
    for i in range(ndim):
        if init_map[i] is not None:
            startPoint.append(init_map[i])
        else:
            startPoint.append(0.0)  # fallback default
    # --------------------------------------------------------------

    #-------------------
    #  SET STEP SIZE
    #-------------------
    if stepSize is None:
        stepSize = [0.01] * ndim

    #-------------------
    #  SET CONFIDENCE LEVEL FOR 4D CASE 90% CL
    #-------------------
    errordef = 7.78

    # import scipy
    # import scipy.stats
    # print(scipy.stats.chi2.ppf(0.9 , df=4))

    # cl_to_errordef_4d = {
    #     68.3: 4.72,
    #     90.0: 7.78,
    #     95.0: 9.49,
    #     99.0: 13.28
    # }

    
    #-------------------
    #  CREATE MINIMIZER
    #-------------------

    minimizer = ROOT.Math.Factory.CreateMinimizer(minimizerName, algoName)
    if not minimizer:
        raise RuntimeError(f"Cannot create minimizer \"{minimizerName}\"")
    

    #-------------------
    #  SET OPTIONS
    #-------------------

    minimizer.SetMaxFunctionCalls(maxFunctionCalls)
    minimizer.SetMaxIterations(maxIterations)
    minimizer.SetTolerance(tolerance)
    minimizer.SetPrintLevel(printLevel)
    minimizer.SetErrorDef(errordef)
    f = ROOT.Math.Functor(func, ndim)
    minimizer.SetFunction(f)

    
    variable = list(startPoint)

    #-------------------
    #  SET PARAMETERS
    #-------------------

    # renaming variable names to match model parameters and set parameters 
    for i in range(ndim):
        if i < len(param_names):
            name = param_names[i]
        else:
            name = f"x{i}"
        minimizer.SetVariable(i, name, variable[i], stepSize[i])

    #could replace the code above by the following code to set parameters without renaming

    # for i in range(ndim):
    #     minimizer.SetVariable(i, f"x{i}", variable[i], stepSize[i])


    #-------------------
    #  RUN MINIMIZATION
    #-------------------

    minimization = minimizer.Minimize()
    if not minimization:
        return {'success': False}
    

    #-------------------
    # GET HESSE ERROR
    #-------------------

    # Create empty arrays to store the results
    xs = np.zeros(ndim)           # parameter values at minimum
    hesse_errors = np.zeros(ndim) # symmetric Hesse errors

    # Loop over each parameter and extract the value and Hesse error
    for i in range(ndim):
        xs[i] = minimizer.X()[i]          # get the fitted value of parameter i
        hesse_errors[i] = minimizer.Errors()[i]  # get the Hesse error for parameter i


    #-------------------
    # GET MINOS ERROR
    #-------------------
    
    # Initialize arrays to store MINOS errors
    minos_errors_low = np.zeros(ndim)
    minos_errors_up = np.zeros(ndim)

    # Temporary arrays for ROOT's GetMinosError
    errLow = np.zeros(1, dtype=np.float64)
    errUp  = np.zeros(1, dtype=np.float64)

    for i in range(ndim):
        success = minimizer.GetMinosError(i, errLow, errUp)
        if success:
            minos_errors_low[i] = errLow[0]
            minos_errors_up[i] = errUp[0]
        else:
            # fallback to Hesse errors if MINOS fails
            minos_errors_low[i] = -hesse_errors[i]
            minos_errors_up[i] = hesse_errors[i]



    # print results
    print("\nMinimization results (values ± Hesse ± MINOS):")
    for i in range(ndim):
        print(f"{param_names[i]}: {xs[i]:.6f} "
              f"± {hesse_errors[i]:.6f} "
              f"[{minos_errors_low[i]:+.6f}, {minos_errors_up[i]:+.6f}]")

    print(f"\nStatus: {minimizer.Status()} (0 = success)\n")
    # ----------------------

    return {
        'success': minimization and minimizer.Status() == 0,
        'x': xs,
        'status': minimizer.Status(),
        'hesse_errors': hesse_errors,
        'minos_errors_low': minos_errors_low,
        'minos_errors_up': minos_errors_up,
    }


# 4D Test Function
def StyblinskiTang4D(vecx):
    result = 0.0
    for i in range(4):
        x = vecx[i]
        result += x**4 - 16*x**2 + 5*x
    return result / 2.0


if __name__ == "__main__":
    # Explicit initial guesses
    result = GenericMinimization(
        func=StyblinskiTang4D,
        ndim=4,
        mg_init=-1.0,
        eps_init=-1.5,
        a1_init=-2.0,
        a2_init=-2.5,
        confidenceLevel=90.0,
        printLevel=0,  # quiet ROOT
    )
