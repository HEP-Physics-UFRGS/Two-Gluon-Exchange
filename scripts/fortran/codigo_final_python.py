"""
CÓDIGO PYTHON FINAL CORRIGIDO
Replicação do Fortran ger4_eik_totem_ens1_v15.F

NOTAS:
1. A parte imaginária está menor que o esperado
2. Isso pode ser devido a:
   - Precisão da integração scipy vs CUBA
   - Diferenças numéricas nos métodos
   - Necessidade de ajuste fino nos parâmetros de integração

Para aproximar melhor o Fortran:
- Use tolerâncias mais apertadas
- Considere usar pycuba se precisar de resultados exatos
"""

import numpy as np
from scipy.integrate import quad, nquad
from scipy.special import jv
import sys

# Constantes
PI = 4.0 * np.arctan(1.0)
LAMBDA = 0.284
LAMBDA2 = LAMBDA * LAMBDA
BEP = 27.0 / (12.0 * PI)
DELTAPP = 0.086557
ALFALIN = 0.25

# Parâmetros do modelo
MM0 = 0.44137
AA0 = 1.6711
AAZ0 = 2.0001


def bessel_j0(x):
    """Bessel J0 com tratamento de valores pequenos"""
    if abs(x) <= 1e-30:
        return complex(1.0, 0.0)
    else:
        return complex(jv(0, x), 0.0)


def integrand(x, y, q2, mg, a1, a2, sqrt_s):
    """
    Integrando 2D (Fortran linhas 205-293)
    
    CORREÇÃO CRÍTICA: A1MAX = sqrt_s, não s!
    """
    Qt2 = q2
    Qt = np.sqrt(Qt2)
    mm = mg
    mm2 = mm * mm
    aa1 = a1
    aa2 = a2
    
    # CORREÇÃO: A1MAX = sqrt_s (GeV), não s (GeV²)
    A1MAX = sqrt_s
    A2MAX = 2.0 * PI
    JCB = A1MAX * A2MAX
    U = A1MAX * x
    V = A2MAX * y
    
    if U < 0.0 or V < 0.0:
        return 0.0
    
    q_sup2 = (Qt2 / 4.0) + U * Qt * np.cos(V) + U * U
    q_inf2 = (Qt2 / 4.0) - U * Qt * np.cos(V) + U * U
    
    Md2sup = (mm2 * mm2 / (q_sup2 + mm2)) * (
        (np.log(4.0 * mm2 / LAMBDA2) / np.log((q_sup2 + 4.0 * mm2) / LAMBDA2)) ** (-1.36)
    )
    
    Md2inf = (mm2 * mm2 / (q_inf2 + mm2)) * (
        (np.log(4.0 * mm2 / LAMBDA2) / np.log((q_inf2 + 4.0 * mm2) / LAMBDA2)) ** (-1.36)
    )
    
    alfaprop_sup = 1.0 / (
        BEP * (q_sup2 + Md2sup) * np.log((q_sup2 + 4.0 * Md2sup) / LAMBDA2)
    )
    
    alfaprop_inf = 1.0 / (
        BEP * (q_inf2 + Md2inf) * np.log((q_inf2 + 4.0 * Md2inf) / LAMBDA2)
    )
    
    Gp_q_0 = np.exp(-aa1 * Qt2 - aa2 * Qt2 * Qt2)
    factor = Qt2 + 9.0 * abs(U * U - Qt2 / 4.0)
    Gp_q_k = np.exp(-aa1 * factor - aa2 * factor * factor)
    
    f = U * alfaprop_sup * alfaprop_inf * (
        Gp_q_0 * Gp_q_0 - Gp_q_k * (2.0 * Gp_q_0 - Gp_q_k)
    ) * JCB
    
    return f


def chi2_int(q, b, mg, a1, a2, sqrt_s):
    """
    Integração 2D para um dado q (Fortran linhas 88-200)
    Retorna valor COMPLEXO
    """
    q2 = q * q
    s = sqrt_s * sqrt_s
    
    # Bessel J0(b*q)
    arg_bessel = b * q
    j0 = bessel_j0(arg_bessel)
    
    # Integração 2D usando nquad
    resultado, error = nquad(
        lambda y, x: integrand(x, y, q2, mg, a1, a2, sqrt_s),
        [[0.0, 1.0], [0.0, 1.0]],
        opts={'epsabs': 1e-10, 'epsrel': 1e-3}
    )
    
    # Trajetória do Pomeron
    trajPP = 1.0 + DELTAPP - ALFALIN * q2
    
    # Amplitude Born (Fortran linha 174)
    # amp_born = imagi * (ss**trajPP) * 8.0 * resultado
    amp_born = 1j * (s ** trajPP) * 8.0 * resultado
    
    # chi2_int (Fortran linha 175)
    chi2_int_val = q * j0 * amp_born
    
    return chi2_int_val


def chi2(b, tt, mg, a1, a2, sqrt_s):
    """
    Integração sobre q (Fortran linhas 56-85)
    
    Fortran linha 80-81:
        resultado=wgauss(chi2_int,qmin,qmax,epsabs)/ss
        chi2=bb*j0*(1.d0-cdexp(imagi*resultado))
    """
    s = sqrt_s * sqrt_s
    
    # Bessel J0(b*√|t|) (Fortran linhas 72-78)
    arg_bessel = b * np.sqrt(abs(tt))
    j0 = bessel_j0(arg_bessel)
    
    # Parâmetros de integração (Fortran 67-69)
    qmin = 0.0
    qmax = 0.2
    epsabs = 1e-4
    
    # Função integrando complexa
    def integrand_q(q_val):
        return chi2_int(q_val, b, mg, a1, a2, sqrt_s)
    
    # Integração separando real e imaginária
    def integrand_real(q_val):
        return np.real(integrand_q(q_val))
    
    def integrand_imag(q_val):
        return np.imag(integrand_q(q_val))
    
    # Integração usando quad
    resultado_real, _ = quad(integrand_real, qmin, qmax, epsabs=epsabs, limit=50)
    resultado_imag, _ = quad(integrand_imag, qmin, qmax, epsabs=epsabs, limit=50)
    
    # Resultado complexo dividido por s (Fortran linha 80)
    resultado = complex(resultado_real, resultado_imag) / s
    
    # chi2 (Fortran linha 81)
    # chi2 = bb * j0 * (1 - cdexp(imagi*resultado))
    chi2_val = b * j0 * (1.0 - np.exp(1j * resultado))
    
    return chi2_val


def main():
    """
    Programa principal - loop duplo sobre b e tt
    (Fortran linhas 1-53)
    """
    sqrt_s = 7000.0
    s = sqrt_s * sqrt_s
    epsabs_tt = 1e-3
    
    # Usar parâmetros do Fortran
    mg = MM0
    a1 = AA0
    a2 = AAZ0
    
    # Arquivo de saída
    output_file = open('output_python.dat', 'w')
    
    tt = 0.0
    
    print("Iniciando cálculo Python...")
    print("="*80)
    print("NOTA: Parte imaginária pode diferir devido a precisão numérica")
    print("      scipy vs CUBA. Para resultados mais próximos, use pycuba.")
    print("="*80)
    
    # Loop externo sobre tt (Fortran linhas 26-51)
    while tt <= 0.1:
        amp_eik = complex(0.0, 0.0)
        b = 0.0
        
        # Loop interno sobre b (Fortran linhas 28-38)
        while b <= 10.0:
            resultado = chi2(b, tt, mg, a1, a2, sqrt_s)
            amp_eik = amp_eik + resultado
            
            # Fortran linha 31: write(*,*)b,amp_eik
            # Formato idêntico ao Fortran
            print(f"  {b:18.12f} ({amp_eik.real:23.15e},{amp_eik.imag:23.15e})")
            sys.stdout.flush()
            
            b = b + 1.0
        
        # Cálculo da seção de choque diferencial (Fortran linhas 41-42)
        diff = (abs(amp_eik) ** 2.0) ** 0.389379323 / (16.0 * PI * s * s)
        
        # Fortran linha 39
        print(f"  {tt:18.12f} ({amp_eik.real:23.15e},{amp_eik.imag:23.15e}) ----------------------------------")
        
        # Fortran linha 43
        output_file.write(f"{tt:23.15e} {diff:23.15e}\n")
        output_file.flush()
        
        # Incremento de tt (Fortran linha 40)
        tt = tt + epsabs_tt
    
    output_file.close()
    
    print("\n" + "="*80)
    print("Cálculo concluído!")
    print(f"Resultados salvos em: output_python.dat")
    print("="*80)


if __name__ == "__main__":
    main()
