import numpy as np
from scipy.integrate import fixed_quad
import os
import plotly.graph_objects as go  # Alterado de matplotlib para plotly
import pandas as pd


# Leitura do arquivo com separação por espaços
data_atlas = pd.read_csv(
    "data/sigma_tot/ensemble_atlas.dat",
    delim_whitespace=True,
    header=None,
    nrows=69  # lê apenas as 70 primeiras linhas
)

x_atlas = data_atlas[0].to_numpy()
y_atlas = data_atlas[1].to_numpy()
y_error_atlas = data_atlas[2].to_numpy()


data_totem = pd.read_csv(
    "data/sigma_tot/ensemble_totem.dat",
    delim_whitespace=True,
    header=None,
    nrows=84  # lê apenas as 70 primeiras linhas
)
x_totem = data_totem[0].to_numpy()
y_totem = data_totem[1].to_numpy()
y_error_totem = data_totem[2].to_numpy()


# === Global Configuration and Constants ===
start_sqrt_s = 1  # Global parameter controlling energy scale
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


#minimized
epsilon_atlas = 0.061
epsilon_totem = 0.078

model_params = {
    'atlas': {
        'log': {
            'mg': 0.334,
            'epsilon': 0.061,
            'a1': 1.604,
            'a2': 3.044
        },
        'pl': {
            'mg': 0.389,
            'epsilon': 0.061,
            'a1': 1.495,
            'a2': 2.161
        }
    },
    'totem': {
        'log': {
            'mg': 0.363,
            'epsilon': 0.079,
            'a1': 1.63,
            'a2': 3.28
        },
        'pl': {
            'mg': 0.424,
            'epsilon': 0.0775,
            'a1': 1.454,
            'a2': 2.93
        }
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
    line_styles = ['solid', 'dot', 'dash', 'dashdot']

    fig = go.Figure()  # Criar figura do Plotly em vez de matplotlib

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
        
        # Adicionar traço ao gráfico do Plotly
        fig.add_trace(go.Scatter(
            x=sqrt_s_lst,
            y=sigma_tot_lst,
            mode='lines+markers',
            line=dict(
                color=line_color,
                dash=line_styles[i],
                width=1
            ),
            marker=dict(
                size=3
            ),
            name=f'{mass_model} {ensemble}'
        ))

    fig.add_trace(go.Scatter(
        x=x_atlas,
        y=y_atlas,
        mode='markers',
        marker=dict(
            color='black',
            size=6,
            symbol='square'
        ),
        error_y=dict(
            type='data',
            array=y_error_atlas,
            visible=True
        ),
        name='ATLAS Data'
    ))

    fig.add_trace(go.Scatter(
        x=x_totem,
        y=y_totem,
        mode='markers',
        marker=dict(
            color='black',
            size=6,
            symbol='circle'
        ),
        error_y=dict(
            type='data',
            array=y_error_totem,
            visible=True
        ),
        name='TOTEM Data'
    ))


    # Configurar layout do gráfico
    fig.update_layout(
    title='Sigma Tot vs. sqrt(s) all models',
    xaxis=dict(
        title='sqrt(s) [GeV]',
        type='log',
        # range=[np.log10(2000), np.log10(15000)],
    ),
    yaxis=dict(
        title='Sigma Tot [mb]',
        # range=[80, 120]
    ),
    showlegend=True,
    legend=dict(
        title='Modelo'
    ),
    plot_bgcolor='white',
    hovermode='x unified'
)
    
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')

    fig.show(renderer="browser")
    # fig.write_html("results/sigma_tot/sigma_tot_minimized_with_data.html")
    # fig.write_image("results/sigma_tot/sigma_tot_minimized_with_data.pdf", width=1200, height=600)

if __name__ == "__main__":
    main()