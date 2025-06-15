import numpy as np
from scipy.integrate import fixed_quad
from scipy.integrate import quad

import os
import plotly.graph_objects as go

# === Global Configuration and Constants ===
start_q2 = 0.006  # Start of q2 range
max_q2 = 0.2   # End of q2 range
q2_step = 0.005


b_0 = (33 - 6) / (12 * np.pi)  # β0 for nf=3
Lambda = 0.284  # ΛQCD in GeV
gamma_1 = 0.084
gamma_2 = 2.36
rho = 4.0
epsilon_atlas = 0.0753
epsilon_totem = 0.0892

dif_sigma_lst_log = []
dif_sigma_lst_pl = []


q2_lst = []
error_lst = []

s0 = 1.0 
alpha_prime = 0.25
sqrt_s = 7000

model_params = {
    'atlas': {
        'log': {'mg': 0.356, 'a1': 1.373, 'a2': 2.5},
        'pl':  {'mg': 0.421, 'a1': 1.517, 'a2': 2.05}
    },
    'totem': {
        'log': {'mg': 0.380, 'a1': 1.491, 'a2': 2.77},
        'pl':  {'mg': 0.447, 'a1': 1.689, 'a2': 1.70}
    }
}

epsilon_values = {
    'atlas': epsilon_atlas,
    'totem': epsilon_totem
}

# === Auxiliary Functions for Physical Model ===
def save_results(filename, q2_val, diff_T, real_part, im_part, sigma_tot):
    header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, 'a') as file:
        if header:
            file.write(f"{'q2':<15} {'Diff_T':<15} {'Real part':<15} {'Imag part':<15} {'Sigma_Tot':<15}\n")
        file.write(f"{q2_val:<15.6f} {diff_T:<15.8f} {real_part:<15.8f} {im_part:<15.8f} {sigma_tot:<15.8f}\n")

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
    # print(f'alpha_d = {(b_0 * (q2 + m2) * np.log((q2 + 4 * m2) / (Lambda ** 2)))}')

    return 1.0 / (b_0 * (q2 + m2) * np.log((q2 + 4 * m2) / (Lambda ** 2)))

def T_1(k, q, phi, mg, a1, a2, m2_func):

    q2 = q 
    
    qk_cos = np.sqrt(q) * k * np.cos(phi)
    qk_plus_squared = q2 / 4 + qk_cos + k ** 2
    qk_minus_squared = q2 / 4 - qk_cos + k ** 2

    alpha_D_plus = alpha_D(qk_plus_squared, mg, m2_func)
    alpha_D_minus = alpha_D(qk_minus_squared, mg, m2_func)
    G0 = G_p(q2, a1, a2)

    # m2 = m2_func(q2, mg)
    # G = ((4 * (m2 ** 2) + 2.79 * q2)/(4 * m2 ** 2 + q2)) * 1/((1+(q2/0.71))**2)

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

    # m2 = m2_func(q2, mg)
    # Gz = ((4 * (m2 ** 2) + 2.79 * q2)/(4 * m2 ** 2 + q2)) * 1/((1+(q2/0.71))**2)
    # Gm = ((4 * m2 ** 2 + 2.79 * factor) / (4 * m2 ** 2 + factor)) * 1/((1 + (factor/0.71))**2)
    # print(f'Gm = {Gm}, q2 = {q2}, Gz = {Gz}, factor = {factor}')
    # print(f'fator = {(2 * Gz - Gm) * Gm}')

    return alpha_D_plus * alpha_D_minus * G_minus * (2 * G0 - G_minus)

def integrand(y, x, mg, a1, a2, m2_func, q_val):

    k = sqrt_s * x 
    phi = 2 * np.pi * y
    jacobian = 2 * np.pi * sqrt_s 

    # print(f't1 = {T_1(k, q_val, phi, mg, a1, a2, m2_func)} , t2 = {T_2(k, q_val, phi, mg, a1, a2, m2_func)}')

    return k * (T_1(k, q_val, phi, mg, a1, a2, m2_func) - T_2(k, q_val, phi, mg, a1, a2, m2_func)) * jacobian 

def amp_calculation(diff_T, s, epsilon, t):

    alpha_pomeron = 1.0 + epsilon + alpha_prime * t
    regge_factor = (s**alpha_pomeron) * 1/(s0**(alpha_pomeron-1))

    return 1j * 8 * regge_factor * diff_T  

def sigma_tot(amp_value, s):
    return amp_value.imag / s * 0.389379323

def differential_sigma(amp_value, s):

    amp_squared = amp_value.imag * amp_value.imag
    denominator  =  (16 * np.pi * s**2)

    # print(f'amplitude = {ampli:.10e}, denominator = {denominator}, s = {s}')

    return  amp_squared / denominator * 0.389379323

    

global q2


# === Main Function ===
def main():
    mass_model_log = 'log'
    mass_model_pl = 'pl'

    ensemble = 'atlas'

    n_points = 10000

    m2_func_log = get_m2_function(mass_model_log)
    m2_func_pl = get_m2_function(mass_model_pl)

    epsilon_atlas = 0.0753

    mg_log_atlas = 0.356
    a1_log_atlas = 1.373
    a2_log_atlas = 2.50

    mg_pl_atlas = 0.421
    a1_pl_atlas = 1.517
    a2_pl_atlas = 2.05

    sqrt_s_values = [7000, 8000, 13000]
    scale_factors = {7000: 1, 8000: 10, 13000: 100}

    fig = go.Figure()

    for sqrt_s in sqrt_s_values:
        scale = scale_factors[sqrt_s]

        dif_sigma_lst_log = []
        dif_sigma_lst_pl = []
        q2_lst = []

        print(f"=== Starting calculation for sqrt(s) = {sqrt_s/1000} TeV ===")

        q2 = start_q2 
        while q2 <= max_q2: 
            t = -q2

            def inner_integral_log(x):
                return fixed_quad(
                    lambda y: integrand(y, x, mg_log_atlas, a1_log_atlas, a2_log_atlas, m2_func_log, q2),
                    0, 1,
                    n=n_points
                )[0]

            integral_value_log = fixed_quad(
                inner_integral_log,
                0, 1,
                n=n_points
            )[0]

            diff_T_log = integral_value_log
            s = sqrt_s ** 2
            amp_value_log = amp_calculation(diff_T_log, s, epsilon_atlas, t)
            dif_sigma_value_log = differential_sigma(amp_value_log, s) * scale
            dif_sigma_lst_log.append(dif_sigma_value_log)

            def inner_integral_pl(x):
                return fixed_quad(
                    lambda y: integrand(y, x, mg_pl_atlas, a1_pl_atlas, a2_pl_atlas, m2_func_pl, q2),
                    0, 1,
                    n=n_points
                )[0]

            integral_value_pl = fixed_quad(
                inner_integral_pl,
                0, 1,
                n=n_points
            )[0]

            diff_T_pl = integral_value_pl
            amp_value_pl = amp_calculation(diff_T_pl, s, epsilon_atlas, t)
            dif_sigma_value_pl = differential_sigma(amp_value_pl, s) * scale
            dif_sigma_lst_pl.append(dif_sigma_value_pl)

            q2_lst.append(q2)
            q2 += q2_step

        print(f"=== Completed for sqrt(s) = {sqrt_s/1000} TeV ===")

        fig.add_trace(go.Scatter(
            x=q2_lst,
            y=dif_sigma_lst_log,
            mode='markers',
            name=f'log √s={sqrt_s//1000} TeV ×{scale}',
            marker=dict(size=5),
            line=dict(width=2)
        ))

        fig.add_trace(go.Scatter(
            x=q2_lst,
            y=dif_sigma_lst_pl,
            mode='markers',
            name=f'pl √s={sqrt_s//1000} TeV ×{scale}',
            marker=dict(size=5),
            line=dict(width=2, dash='dash')
        ))

    fig.update_layout(
        title=f'Differential cross section vs. |t| for log and pl models ({ensemble})',
        xaxis_title='|t| (GeV²)',
        yaxis_title='dσ/dt (mb/GeV²)',
        yaxis_type='log',
        showlegend=True
    )
    print(dif_sigma_lst_log)
    print(dif_sigma_lst_pl)

    plot_filename = f"differential_cross_section_plot_{ensemble}_for_7_8_13_TeV.html"
    fig.write_html(plot_filename)
    # fig.write_image(f"differential_cross_section_plot_{ensemble}_for_7_8_13_TeV.pdf")
    print(f"Plot saved to {plot_filename}")
    fig.show(renderer="browser")
    



if __name__ == "__main__":
    main()
