"""
Tentativa: usar k como momento transverso (0 a sqrt_s)
ao invés de U = s * x
"""
import numpy as np
from scipy.integrate import dblquad

# Parâmetros do Fortran
b_0 = (33 - 6) / (12 * np.pi)
Lambda = 0.284
gamma_2 = 2.36
rho = 4.0
mg = 0.44137
a1 = 1.6711
a2 = 2.0001
sqrt_s = 7000.0
s = sqrt_s * sqrt_s

def m2_pl(q2):
    lambda_squared = Lambda ** 2
    rho_mg_squared = rho * mg ** 2
    ratio = np.log((q2 + rho_mg_squared) / lambda_squared) / np.log(rho_mg_squared / lambda_squared)
    return (mg ** 4 / (q2 + mg ** 2)) * ratio ** (gamma_2 - 1)

def alpha_D(q2):
    m2 = m2_pl(q2)
    return 1.0 / (b_0 * (q2 + m2) * np.log((q2 + 4 * m2) / (Lambda ** 2)))

def integrand_with_k(x, y, q2):
    """
    Usando k = sqrt_s * x (momento transverso em GeV)
    """
    k = sqrt_s * x  # momento transverso: 0 to sqrt_s GeV
    phi = 2 * np.pi * y  # ângulo: 0 to 2π
    jacobian = sqrt_s * 2 * np.pi  # d k d phi
    
    Qt2 = q2
    Qt = np.sqrt(Qt2)
    
    # q_sup2 e q_inf2 usando k (não U)
    q_sup2 = (Qt2/4.0) + k * Qt * np.cos(phi) + k * k
    q_inf2 = (Qt2/4.0) - k * Qt * np.cos(phi) + k * k
    
    # Running coupling
    try:
        alfaprop_sup = alpha_D(q_sup2)
        alfaprop_inf = alpha_D(q_inf2)
    except:
        return 0.0
    
    if not (np.isfinite(alfaprop_sup) and np.isfinite(alfaprop_inf)):
        return 0.0
    
    # Form factors
    Gp_q_0 = np.exp(-a1 * Qt2 - a2 * Qt2 * Qt2)
    factor = Qt2 + 9.0 * abs(k*k - Qt2/4.0)  # usando k, não U
    Gp_q_k = np.exp(-a1 * factor - a2 * factor * factor)
    
    # Integrando final
    f = k * alfaprop_sup * alfaprop_inf * (Gp_q_0*Gp_q_0 - Gp_q_k*(2.0*Gp_q_0 - Gp_q_k)) * jacobian
    
    return f
def compute_amplitude(q, diff_t):
    """
    Replica exatamente o LEVEL 3 do Fortran:
    amp_born = i * (s**trajPP) * 8 * diff_t
    """
    deltaPP = 0.086557
    alfalin = 0.25

    q2 = q * q
    trajPP = 1.0 + deltaPP - alfalin * q2

    amp_born = 1j * (s ** trajPP) * 8.0 * diff_t
    return amp_born

# Teste
print("=" * 80)
print("TESTE: Usando k = sqrt_s * x (momento em GeV)")
print("=" * 80)

q_values = np.linspace(0.01, 0.200, 50)
fortran_values = {
    0.01: 6.8824395154,
    0.02: 6.8717385387,
    0.03: 6.8539303612,
    0.04: 6.8290551842
}

for q in q_values:
    q2 = q * q
    
    result, error = dblquad(
        lambda y, x: integrand_with_k(x, y, q2),
        0, 1,
        0, 1,
        epsabs=1e-10,
        epsrel=1e-6
    )
    
    # diff_t_fortran = fortran_values[q]
    
    print(f"\nq = {q:.3f} (q² = {q2:.6f})")
    print(f"  Python (k=sqrt_s*x): diff_t = {result:12.6f} ± {error:.2e}")
    amp_born = compute_amplitude(q, result)

    print(f"  amp_born = {amp_born}")
    print(f"  |amp_born| = {abs(amp_born)}")

    # print(f"  Fortran (U=s*x):     diff_t = {diff_t_fortran:12.6f}")
    # print(f"  Diferença:                    {result - diff_t_fortran:12.6f}")
    # print(f"  Razão:                        {result / diff_t_fortran if diff_t_fortran != 0 else 0:.4f}")