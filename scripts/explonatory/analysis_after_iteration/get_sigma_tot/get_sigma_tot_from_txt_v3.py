import numpy as np
from scipy.integrate import fixed_quad
import os
import glob

# === Parâmetros fixos ===
b_0 = (33 - 6) / (12 * np.pi)
Lambda = 0.284
gamma_1 = 0.084
gamma_2 = 2.36
rho = 4.0
s0 = 1.0
sqrt_s = 13000
n_points = 10000

# === Funções físicas ===
def m2_log(q2, mg):
    # lambda_squared = Lambda ** 2
    # rho_mg_squared = rho * mg ** 2
    # ratio = np.log((q2 + rho_mg_squared) / lambda_squared) / np.log(rho_mg_squared / lambda_squared)
    # return mg ** 2 * ratio ** (-1 - gamma_1)

# for pl, remember to alter the files 
    lambda_squared = Lambda ** 2
    rho_mg_squared = rho * mg ** 2
    ratio = np.log((q2 + rho_mg_squared) / lambda_squared) / np.log(rho_mg_squared / lambda_squared)
    return (mg ** 4 / (q2 + mg ** 2)) * ratio ** (gamma_2 - 1)

def G_p(q2, a1, a2):
    return np.exp(-(a1 * q2 + a2 * q2 ** 2))

def alpha_D(q2, mg, m2_func):
    m2 = m2_func(q2, mg)
    return 1.0 / (b_0 * (q2 + m2) * np.log((q2 + 4 * m2) / (Lambda ** 2)))

def T_1(k, q, phi, mg, a1, a2, m2_func):
    q2 = q ** 2
    qk_cos = q * k * np.cos(phi)
    qk_plus_squared = q2 / 4 + qk_cos + k ** 2
    qk_minus_squared = q2 / 4 - qk_cos + k ** 2

    alpha_D_plus = alpha_D(qk_plus_squared, mg, m2_func)
    alpha_D_minus = alpha_D(qk_minus_squared, mg, m2_func)
    G0 = G_p(q2, a1, a2)
    return alpha_D_plus * alpha_D_minus * G0 ** 2

def T_2(k, q, phi, mg, a1, a2, m2_func):
    q2 = q ** 2
    qk_cos = q * k * np.cos(phi)
    qk_plus_squared = q2 / 4 + qk_cos + k ** 2
    qk_minus_squared = q2 / 4 - qk_cos + k ** 2

    alpha_D_plus = alpha_D(qk_plus_squared, mg, m2_func)
    alpha_D_minus = alpha_D(qk_minus_squared, mg, m2_func)

    factor = q2 + 9 * abs(k ** 2 - q2 / 4)
    G0 = G_p(q2, a1, a2)
    G_minus = G_p(factor, a1, a2)
    return alpha_D_plus * alpha_D_minus * G_minus * (2 * G0 - G_minus)

def integrand(y, x, mg, a1, a2, m2_func):
    k = sqrt_s * x
    phi = 2 * np.pi * y
    jacobian = 2 * np.pi * sqrt_s
    return k * (T_1(k, 0.0, phi, mg, a1, a2, m2_func) - T_2(k, 0.0, phi, mg, a1, a2, m2_func)) * jacobian

def amp_calculation(diff_T, s, epsilon):
    alpha_pomeron = 1.0 + epsilon
    regge_factor = (s / s0) ** alpha_pomeron
    return 1j * 8.0 * regge_factor * diff_T

def sigma_tot(amp_value, s):
    return amp_value.imag / s * 0.389379323

# === Função de extração de valores ===
def extrair_valores(linha):
    partes = [p.strip() for p in linha.split('|')]
    if len(partes) < 9:
        return None
    try:
        down = float(partes[0])
        strategy = float(partes[2])
        tol = float(partes[3])
        ncall = int(partes[4])
        mg = float(partes[5].split('±')[0].strip())
        eps = float(partes[6].split('±')[0].strip())
        a1 = float(partes[7].split('±')[0].strip())
        a2 = float(partes[8].split('±')[0].strip())
        return down, strategy,tol, ncall, mg, eps, a1, a2
    except Exception as e:
        print("Erro ao extrair valores:", e)
        print("Linha problemática:", linha)
        return None

# === Caminhos ===
input_dir = 'results/all_possible_iterations/v7/all_possible_iterations_pl_atlas_run7'
output_dir = os.path.join(input_dir, 'sigma_tot')
os.makedirs(output_dir, exist_ok=True)

# === Processamento de todos os arquivos ===
arquivos = glob.glob(os.path.join(input_dir, '*.txt'))

for caminho_entrada in arquivos:
    nome_arquivo = os.path.basename(caminho_entrada)
    caminho_saida = os.path.join(output_dir, f'output_{nome_arquivo}')

    with open(caminho_entrada, 'r') as arq, open(caminho_saida, 'w') as saida:
        saida.write("down | strategy | tol | ncall | sigma_tot\n")
        for linha in arq:
            linha = linha.strip()
            if not linha or linha.startswith(('Otimização', 'Configuração', '===')):
                continue

            dados = extrair_valores(linha)
            if not dados:
                continue

            down, strategy, tol, ncall, mg, eps, a1, a2 = dados
            m2_func = m2_log
            epsilon = eps
            s = sqrt_s ** 2

            try:
                def inner_integral(x):
                    return fixed_quad(lambda y: integrand(y, x, mg, a1, a2, m2_func), 0, 1, n=n_points)[0]

                integral_value = fixed_quad(inner_integral, 0, 1, n=n_points)[0]
                amp = amp_calculation(integral_value, s, epsilon)
                sigma = sigma_tot(amp, s)
                saida.write(f"{down:.2f} | {strategy:.2f} | {tol} | {ncall} | {sigma:.3f}\n")
            except Exception as e:
                print(f"Erro no cálculo para linha: {linha}")
                print(e)
