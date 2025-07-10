import numpy as np
from scipy.integrate import fixed_quad
b_0 = (33 - 6) / (12 * np.pi)
Lambda = 0.284  # ΛQCD in GeV
gamma_1 = 0.084
gamma_2 = 2.36
rho = 4.0
s0 = 1.0
alpha_prime = 0.25
# Constantes
s0 = 1.0  # Valor de referência para o fator de Regge
n_points = 10  # Número de pontos para a quadratura fixa

def extrair_configuracoes(arquivo_entrada):
    """Extrai as configurações do arquivo de entrada"""
    configuracoes = []
    with open(arquivo_entrada, 'r') as f:
        for linha in f:
            if '|' not in linha or '±' not in linha:
                continue
                
            partes = linha.split('|')
            
            try:
                down = partes[0].strip()
                up = partes[1].strip()
                strategy = partes[2].strip()
                ncall = partes[3].strip()
                
                mg = float(partes[4].strip().split('±')[0])
                eps = float(partes[5].strip().split('±')[0])
                a1 = float(partes[6].strip().split('±')[0])
                a2 = float(partes[7].strip().split('±')[0])
                
                configuracoes.append({
                    'down': down,
                    'up': up,
                    'strategy': strategy,
                    'ncall': ncall,
                    'mg': mg,
                    'eps': eps,
                    'a1': a1,
                    'a2': a2
                })
            except (IndexError, ValueError) as e:
                print(f"Erro ao processar linha: {linha}")
                continue
                
    return configuracoes

def calcular_sigma_tot(epsilon, mg, a1, a2, sqrt_s=13010):
    """Calcula a seção de choque total para uma dada configuração"""
    def integrand(y, x, mg, a1, a2):
        k = sqrt_s * x
        phi = 2 * np.pi * y
        jacobian = 2 * np.pi * sqrt_s
        return k * (T_1(k, 0.0, phi, mg, a1, a2) - T_2(k, 0.0, phi, mg, a1, a2)) * jacobian
    
    def amp_calculation(diff_T, s, epsilon):
        alpha_pomeron = 1.0 + epsilon
        regge_factor = (s / s0) ** alpha_pomeron
        return 1j * 8.0 * regge_factor * diff_T

    s = sqrt_s ** 2

    integral_value = fixed_quad(
        lambda x: fixed_quad(
            lambda y: integrand(y, x, mg, a1, a2),
            0, 1, n=n_points)[0],
        0, 1, n=n_points)[0]
    
    amplitude = amp_calculation(integral_value, s, epsilon)
    return np.imag(amplitude) / s * 0.3894  # Convertendo para milibarns

def G_p(q2, a1, a2):
    return np.exp(-(a1 * q2 + a2 * q2 ** 2))

def alpha_D(q2, mg, m2_func):
    m2 = m2_func(q2, mg)
    return 1.0 / (b_0 * (q2 + m2) * np.log((q2 + 4 * m2) / (Lambda ** 2)))

def T_1(k, q, phi, mg, a1, a2, m2_func):
    q2 = q 
    qk_cos = np.sqrt(q) * k * np.cos(phi)
    qk_plus_squared = q2 / 4 + qk_cos + k ** 2
    qk_minus_squared = q2 / 4 - qk_cos + k ** 2
    alpha_D_plus = alpha_D(qk_plus_squared, mg, m2_func)
    alpha_D_minus = alpha_D(qk_minus_squared, mg, m2_func)
    G0 = G_p(q2, a1, a2)
    return alpha_D_plus * alpha_D_minus * G0 ** 2

def T_2(k, q, phi, mg, a1, a2, m2_func):
    q2 = q 
    qk_cos = np.sqrt(q) * k * np.cos(phi)
    qk_plus_squared = q2 / 4 + qk_cos + k ** 2
    qk_minus_squared = q2 / 4 - qk_cos + k ** 2
    alpha_D_plus = alpha_D(qk_plus_squared, mg, m2_func)
    alpha_D_minus = alpha_D(qk_minus_squared, mg, m2_func)
    factor = q2 + 9 * abs(k ** 2 - q2 / 4)
    G0 = G_p(q2, a1, a2)
    G_minus = G_p(factor, a1, a2)
    return alpha_D_plus * alpha_D_minus * G_minus * (2 * G0 - G_minus)

def processar_configuracoes(arquivo_entrada, arquivo_saida, limite_sigma=100):
    """Processa as configurações e grava apenas as com σ_tot > limite"""
    configuracoes = extrair_configuracoes(arquivo_entrada)
    
    with open(arquivo_saida, 'w') as f_out:
        # Escreve cabeçalho
        f_out.write("DOWN | STRATEGY | NCALL | SIGMA_TOT\n")
        
        for config in configuracoes:
            sigma_tot = calcular_sigma_tot(
                config['eps'],
                config['mg'],
                config['a1'],
                config['a2']
            )
            
            if sigma_tot > limite_sigma:
                linha_saida = (
                    f"{config['down']} | {config['strategy']} | "
                    f"{config['ncall']} | {sigma_tot:.2f}\n"
                )
                f_out.write(linha_saida)
                print(f"Configuração válida encontrada: {linha_saida.strip()}")

# Exemplo de uso
if __name__ == "__main__":
    arquivo_entrada = "results/iteration_over_all_log/resultados_otimizacao_log_atlas_eps_0_mg_0_a2_0.txt"  # Substitua pelo seu arquivo de entrada
    arquivo_saida = "configuracoes_filtradas.txt"
    limite_sigma = 100  # Apenas configurações com σ_tot > 100
    
    processar_configuracoes(arquivo_entrada, arquivo_saida, limite_sigma)
    print(f"Processamento concluído. Resultados salvos em {arquivo_saida}")