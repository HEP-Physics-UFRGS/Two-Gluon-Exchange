"""
Tentativa: usar k como momento transverso (0 a sqrt_s)
ao invés de U = s * x
"""
import numpy as np
from scipy.integrate import dblquad, quad
from scipy.special import j0 as bessel_j0

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

def chi2_int(q, b, tt):
    """
    Replica exatamente a função chi2_int do Fortran (linha 77-159)
    """
    q2 = q * q
    
    # Bessel J0(b*q)
    if b * q <= 1e-30:
        j0 = 1.0 + 0j
    else:
        j0 = bessel_j0(b * q) + 0j
    
    # Integração em 2D para diff_t
    result, error = dblquad(
        lambda y, x: integrand_with_k(x, y, q2),
        0, 1,
        0, 1,
        epsabs=1e-10,
        epsrel=1e-6
    )
    diff_t = result
    
    # Amplitude de Born
    deltaPP = 0.086557
    alfalin = 0.25
    trajPP = 1.0 + deltaPP - alfalin * q2
    amp_born = 1j * (s ** trajPP) * 8.0 * diff_t
    
    # chi2_int = q * j0 * amp_born
    chi2_int_val = q * j0 * amp_born
    
    return chi2_int_val

def chi2(b, tt):
    """
    Replica exatamente a função chi2 do Fortran (linha 47-74)
    """
    # Bessel J0(b*sqrt(tt))
    if b * np.sqrt(tt) <= 1e-30:
        j0 = 1.0 + 0j
    else:
        j0 = bessel_j0(b * np.sqrt(tt)) + 0j
    
    # Integração em q de 0 a 0.2
    qmin = 0.0
    qmax = 0.2
    
    # Calcula chi_val integrando chi2_int
    # Divide resultado final por s conforme linha 71 do Fortran
    def integrand_real(q):
        val = chi2_int(q, b, tt)
        return val.real
    
    def integrand_imag(q):
        val = chi2_int(q, b, tt)
        return val.imag
    
    chi_val_real, _ = quad(integrand_real, qmin, qmax, epsabs=1e-4, epsrel=1e-3, limit=100)
    chi_val_imag, _ = quad(integrand_imag, qmin, qmax, epsabs=1e-4, epsrel=1e-3, limit=100)
    
    chi_val = (chi_val_real + 1j * chi_val_imag) / s
    
    # chi2 = b * j0 * (1 - exp(i*chi_val))
    # Linha 72 do Fortran
    chi2_val = b * j0 * (1.0 - np.exp(1j * chi_val))
    
    return chi2_val

# Teste
epsabs = 1e-3
bmin = 0.0
bmax = 10.0
tt = 0.0

while True:
    print(f" TT:   {tt:22.16f}")
    
    b = 0.0
    amp_eik = 0.0 + 0j
    
    while True:
        print(f"   B:   {b:22.16f}")
        
        resultado = chi2(b, tt)
        
        # Format to match Fortran: "   CHI2:   value              (real,imag)"
        spacing = " " * 14  # Adjust based on Fortran output
        print(f"   CHI2:   {b:22.16f}{spacing}({resultado.real:22.16e},{resultado.imag:22.16e})")
        
        amp_eik = amp_eik + resultado
        
        b = b + 1.0
        if b > 10.0:
            break
    
    tt = tt + epsabs
    if tt > 0.1:
        break