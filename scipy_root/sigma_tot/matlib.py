import numpy as np
from scipy.integrate import fixed_quad
import os
import matplotlib.pyplot as plt
import pandas as pd

# === Global Configuration and Constants ===
start_sqrt_s = 100  # Global parameter controlling energy scale
b_0 = (33 - 6) / (12 * np.pi)  # β0 for nf=3
Lambda = 0.284  # ΛQCD in GeV
gamma_1 = 0.084
gamma_2 = 2.36
rho = 4.0




sigma_tot_lst = []
sqrt_s_lst = []
error_lst = []

s0 = 1.0  # GeV^2

# epsilon_atlas = 0.0753
# epsilon_totem = 0.0892

# model_params = {
#     'atlas': {
#         'log': {'mg': 0.356, 'a1': 1.373, 'a2': 2.5},
#         'pl':  {'mg': 0.421, 'a1': 1.517, 'a2': 2.05}
#     },
#     'totem': {
#         'log': {'mg': 0.380, 'a1': 1.491, 'a2': 2.77},
#         'pl':  {'mg': 0.447, 'a1': 1.689, 'a2': 1.70}
#     }
# }



#minimized parameters for the models
epsilon_atlas = 0.0677
epsilon_totem = 0.080282

model_params = {
    'atlas': {
        'log': {'mg': 0.345, 'a1': 1.5103, 'a2': 2.75},
        'pl':  {'mg': 0.402, 'a1': 1.56, 'a2': 1.97}
    },
    'totem': {
        'log': {'mg': 0.365, 'a1': 1.6401, 'a2': 3.047},
        'pl':  {'mg': 0.455, 'a1': 1.7363, 'a2': 1.87}
    }
}
epsilon_values = {
    'atlas': epsilon_atlas,
    'totem': epsilon_totem
}

# === Auxiliary Functions for Physical Model ===
def save_results(filename, iteration, diff_T, real_part, im_part, sigma_tot):
    header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, 'a') as file:
        if header:
            file.write(f"{'sqrt(s)':<15} {'Diff_T':<15} {'Real part':<15} {'Imag part':<15} {'Sigma_Tot':<15}\n")
        file.write(f"{iteration:<15d} {diff_T:<15.8f} {real_part:<15.8f} {im_part:<15.8f} {sigma_tot:<15.8f}\n")

def m2_log(q2, mg):
    lambda_squared = Lambda ** 2
    rho_mg_squared = rho * mg ** 2
    ratio = np.log((q2 + rho_mg_squared) / lambda_squared) / np.log(rho_mg_squared / lambda_squared)
    return mg ** 2 * ratio ** (-1 - gamma_1)

def m2_pl(q2, mg):
    lambda_squared = Lambda ** 2
    rho_mg_squared = rho * mg ** 2
    ratio = np.log((q2 + rho_mg_squared) / lambda_squared) / np.log(rho_mg_squared / lambda_squared)
    return (mg ** 4 / (q2 + mg ** 2)) * ratio ** (gamma_2 - 1)

def get_m2_function(mass_model):
    return m2_log if mass_model == 'log' else m2_pl

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

# === Main Function ===
def main():
    global start_sqrt_s
    global sqrt_s

    max_sqrt_s = 13000
    step = 100
    n_points = 10000

    scenarios = [
        ('log', 'atlas'),
        ('pl', 'atlas'),
        ('log', 'totem'),
        ('pl', 'totem'),
    ]

    # Estilos de linha diferentes para cada cenário
    line_styles = ['-', ':', '--', '-.']

    plt.figure(figsize=(10, 6))

    for i, (mass_model, ensemble) in enumerate(scenarios):
        sigma_tot_lst = []
        sqrt_s_lst = []

        m2_func = get_m2_function(mass_model)
        params = model_params[ensemble][mass_model]
        mg, a1, a2 = params['mg'], params['a1'], params['a2']
        epsilon = epsilon_values[ensemble]

        sqrt_s = start_sqrt_s
        while sqrt_s <= max_sqrt_s:
            def inner_integral(x):
                return fixed_quad(
                    lambda y: integrand(y, x, mg, a1, a2, m2_func),
                    0, 1,
                    n=n_points
                )[0]

            integral_value = fixed_quad(
                inner_integral,
                0, 1,
                n=n_points
            )[0]

            diff_T = integral_value
            s = sqrt_s * sqrt_s

            amp_value = amp_calculation(diff_T, s, epsilon)
            sigma_tot_value = sigma_tot(amp_value, s)

            sigma_tot_lst.append(sigma_tot_value)
            sqrt_s_lst.append(sqrt_s)

            sqrt_s += step

        # Definir cor baseada no modelo de massa
        line_color = 'red' if mass_model == 'log' else 'blue'
        
        plt.plot(sqrt_s_lst, sigma_tot_lst, 
                linestyle=line_styles[i],
                color=line_color,  # Adicionado a cor aqui
                linewidth=1,
                marker='o',
                markersize=1.5,
                label=f'{mass_model} {ensemble}')

    plt.title('Sigma Tot vs. sqrt(s) all models')
    plt.xlabel('sqrt(s) [GeV]')
    plt.ylabel('Sigma Tot [mb]')
    plt.xscale('log')
    plt.legend(title='Modelo')
    plt.grid(True, which="both", ls="-")

    # output_dir = "plots/"
    # plt.savefig(output_dir + f"cross_section_plot_all_models_lib_with_data_minimized.pdf")
    plt.show()

if __name__ == "__main__":
    main()