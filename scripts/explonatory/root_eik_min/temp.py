import ROOT
import math

def cauchy_principal_value_symmetric():
    """Proper Cauchy principal value integration using symmetry"""
    
    print("PROPER CAUCHY PRINCIPAL VALUE INTEGRATION")
    print("=" * 80)
    
    # For 1/x from -1 to 1, use symmetry: ∫_{-1}^{1} 1/x dx = 0 (principal value)
    # But we need to avoid the singularity at x=0
    
    # Method 1: Split integration and avoid singularity
    def integrate_around_singularity():
        epsilon = 1e-10  # Small offset to avoid singularity
        
        func = ROOT.TF1("f_cauchy", "1/x", -1, 1)
        integ = ROOT.Math.IntegratorOneDim(ROOT.Math.IntegrationOneDim.kADAPTIVESINGULAR)
        
        # Integrate from -1 to -epsilon and from epsilon to 1
        result1 = integ.Integral(func, -1, -epsilon)
        result2 = integ.Integral(func, epsilon, 1)
        total = result1 + result2
        
        return total, result1, result2
    
    # Method 2: Use variable transformation to handle singularity
    def integrate_with_transformation():
        # Transform x = t/(1-t²) or use other variable changes
        # For 1/x, we can use substitution to remove singularity
        pass
    
    # Method 3: Use different integration ranges
    ranges = [
        ([-1, -1e-6, 1e-6, 1], "Fine segmentation"),
        ([-1, -1e-8, 1e-8, 1], "Very fine segmentation"),
        ([-1, -1e-4, 1e-4, 1], "Coarse segmentation")
    ]
    
    methods = {
        'kADAPTIVE': ROOT.Math.IntegrationOneDim.kADAPTIVE,
        'kADAPTIVESINGULAR': ROOT.Math.IntegrationOneDim.kADAPTIVESINGULAR
    }
    
    func = ROOT.TF1("f_cauchy", "1/x", -1, 1)
    
    print("\n1/x from -1 to 1 (Cauchy Principal Value = 0)")
    print("-" * 80)
    
    for range_points, desc in ranges:
        print(f"\n{desc}: {range_points}")
        
        for method_name, method in methods.items():
            integ = ROOT.Math.IntegratorOneDim(method)
            integ.SetRelTolerance(1e-6)
            
            total = 0.0
            status_ok = True
            
            # Integrate each segment
            for i in range(len(range_points) - 1):
                a, b = range_points[i], range_points[i+1]
                try:
                    segment_result = integ.Integral(func, a, b)
                    total += segment_result
                except:
                    status_ok = False
                    break
            
            if status_ok:
                print(f"{method_name:<20} Result: {total:12.6f} Error: {abs(total):10.2e}")
            else:
                print(f"{method_name:<20} FAILED")

def test_singular_function_integration():
    """Test integration of functions with known analytical solutions"""
    
    print("\n\nINTEGRATION OF SINGULAR FUNCTIONS WITH KNOWN SOLUTIONS")
    print("=" * 80)
    
    # Functions that are singular but have finite integrals
    test_cases = [
        {
            'name': '1/sqrt(x) [0,1]',
            'func': ROOT.TF1("f1", "1/sqrt(x)", 0, 1),
            'expected': 2.0,
            'desc': 'Integrable singularity at x=0'
        },
        {
            'name': 'sqrt(x) [0,1]', 
            'func': ROOT.TF1("f2", "sqrt(x)", 0, 1),
            'expected': 2.0/3.0,
            'desc': 'Continuous but derivative singular at x=0'
        },
        {
            'name': 'log(x) [0,1]',
            'func': ROOT.TF1("f3", "log(x)", 0, 1),
            'expected': -1.0,
            'desc': 'Finite but undefined at x=0'
        },
        {
            'name': '1/sqrt(1-x) [0,1]',
            'func': ROOT.TF1("f4", "1/sqrt(1-x)", 0, 0.999),
            'expected': 2.0 - 2.0*math.sqrt(0.001),
            'desc': 'Singularity at x=1'
        }
    ]
    
    methods = {
        'kGAUSS': ROOT.Math.IntegrationOneDim.kGAUSS,
        'kADAPTIVE': ROOT.Math.IntegrationOneDim.kADAPTIVE,
        'kADAPTIVESINGULAR': ROOT.Math.IntegrationOneDim.kADAPTIVESINGULAR
    }
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}: {test_case['desc']}")
        print(f"Expected: {test_case['expected']:.8f}")
        print("-" * 80)
        print(f"{'Method':<20} {'Result':<15} {'Error':<15} {'Status':<10}")
        print("-" * 80)
        
        for method_name, method in methods.items():
            integ = ROOT.Math.IntegratorOneDim(method)
            integ.SetRelTolerance(1e-8)
            integ.SetAbsTolerance(1e-10)
            
            try:
                a = test_case['func'].GetXmin()
                b = test_case['func'].GetXmax()
                result = integ.Integral(test_case['func'], a, b)
                error = abs(result - test_case['expected'])
                status = integ.Status()
                
                print(f"{method_name:<20} {result:<15.8f} {error:<15.2e} {status:<10}")
            except Exception as e:
                print(f"{method_name:<20} FAILED: {str(e)[:30]}")

def adaptive_singularity_handling():
    """Show how to handle singularities adaptively"""
    
    print("\n\nADAPTIVE SINGULARITY HANDLING STRATEGIES")
    print("=" * 80)
    
    # Strategy 1: Variable transformation
    print("\n1. Variable Transformation Method")
    print("-" * 50)
    
    # For ∫₀¹ 1/√x dx, use substitution x = t²
    func_transformed = ROOT.TF1("f_trans", "2", 0, 1)  # After substitution: ∫₀¹ 2 dt
    
    integ = ROOT.Math.IntegratorOneDim(ROOT.Math.IntegrationOneDim.kADAPTIVE)
    result = integ.Integral(func_transformed, 0, 1)
    print(f"Transformed integral: {result:.8f} (Expected: 2.00000000)")
    
    # Strategy 2: Singularity subtraction
    print("\n2. Singularity Subtraction Method")
    print("-" * 50)
    
    # For ∫₀¹ (1/√x) dx, subtract and add the singular part
    def singularity_subtraction():
        # This would be implemented manually
        pass
    
    # Strategy 3: Complex plane integration
    print("\n3. Complex Plane Method (Conceptual)")
    print("-" * 50)
    print("For Cauchy principal value, integrate around pole in complex plane")
    print("Not directly available in ROOT - requires manual implementation")

def practical_recommendations():
    """Provide practical recommendations for singular integrals"""
    
    print("\n\nPRACTICAL RECOMMENDATIONS FOR SINGULAR INTEGRALS")
    print("=" * 80)
    
    recommendations = [
        "1. Use kADAPTIVESINGULAR for integrable singularities",
        "2. Avoid integrating exactly at singular points",
        "3. Split integration domain around singularities",
        "4. Use variable transformations to remove singularities",
        "5. For Cauchy PV, manually split symmetric domains",
        "6. Increase tolerance settings for difficult integrals",
        "7. Check status codes: 0=success, >0=warnings, <0=errors",
        "8. Consider analytical transformation before numerical integration"
    ]
    
    for rec in recommendations:
        print(rec)
    
    print("\nStatus Code Guide:")
    print("0: Success")
    print("1: Maximum iterations reached but result may be OK") 
    print("2: Accuracy not achieved")
    print("3: Invalid input parameters")
    print("4: Integration failed")
    print("5+: Various error conditions")

if __name__ == "__main__":
    cauchy_principal_value_symmetric()
    test_singular_function_integration()
    adaptive_singularity_handling()
    practical_recommendations()