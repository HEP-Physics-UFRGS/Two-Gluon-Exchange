import numpy as np
from scipy.integrate import fixed_quad, quad
import os
import plotly.graph_objects as go
from scipy.special import j0, jv
import pandas as pd
# Leitura do arquivo com separação por espaços
data_atlas = pd.read_csv(
    "data/sigma_tot_2/ensemble_StRh_atlas.dat",
    delim_whitespace=True,
    header=None,
    nrows=70  # lê apenas as 70 primeiras linhas
)

x_atlas = data_atlas[0].to_numpy()
y_atlas = data_atlas[1].to_numpy()
y_error_atlas = data_atlas[2].to_numpy()
# === Global Configuration and Constants ===
start_sqrt_s = 101  # Global parameter controlling energy scale
b_0 = (33 - 6) / (12 * np.pi)  # β0 for nf=3
Lambda = 0.284  # ΛQCD in GeV
gamma_1 = 0.084
gamma_2 = 2.36
rho = 4.0

lst_sigma_tot_born = []
lst_sqrt_s = []
lst_error = []

s0 = 1.0  # GeV^2

epsilon_atlas = 0.0732

max_sqrt_s = 13000
step = 100
n_points = 10000

model_params = {
    'atlas': {
        'pl':  {'mg': 0.417, 'a1': 1.563, 'a2': 2.22}
    }
}

epsilon_values = {
    'atlas': epsilon_atlas
}

lst_amp_born = []
lst_sqrt_s = []
lst_sigma_tot_born = []



q_upper = 0.2
b_upper = 30

epsabs = 1e-10
epsrel = 1e-10
limit = 100_000

# === Auxiliary Functions for Physical Model ===
def m2_pl(q2, mg):
    lambda_squared = Lambda ** 2
    rho_mg_squared = rho * mg ** 2
    ratio = np.log((q2 + rho_mg_squared) / lambda_squared) / np.log(rho_mg_squared / lambda_squared)
    return (mg ** 4 / (q2 + mg ** 2)) * ratio ** (gamma_2 - 1)

def get_m2_function(mass_model):
    return m2_pl

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


def add_iterative_curve(fig, x_data, y_data, 
                        curve_name:str=None, color:str='blue', line_type:str='lines+markers'):

    fig.add_trace(go.Scatter(
    x = x_data,
    y = y_data,
    mode=line_type, 
    name=curve_name,
    line=dict(
        color=color,
        width=2),
    marker=dict(size=4))
)
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')

# Using only PL model with ATLAS
mass_model = 'pl'
ensemble = 'atlas'


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

    lst_sigma_tot_born.append(sigma_tot_value)
    lst_sqrt_s.append(sqrt_s)
    lst_amp_born.append(amp_value)

    sqrt_s += step


lst_s = [val**2 for val in lst_sqrt_s]

fig_sigma = go.Figure()

add_iterative_curve(fig_sigma, lst_sqrt_s, lst_sigma_tot_born, curve_name='sigma tot born')
# Add ATLAS data
fig_sigma.add_trace(go.Scatter(
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

# Configure layout
fig_sigma.update_layout(
    title='Sigma Tot vs. sqrt(s)',
    xaxis=dict(
        title='sqrt(s) [GeV]',
        type='log',
    ),
    yaxis=dict(
        title='Sigma Tot [mb]',
    ),
    showlegend=True,
    legend=dict(
        title='Model/Data'
    ),
    plot_bgcolor='white',
    hovermode='x unified'
)

fig_sigma.update_xaxes(gridcolor='lightgray')
fig_sigma.update_yaxes(gridcolor='lightgray')

fig_sigma.show(renderer = 'browser')
# fig_sigma.write_html('compare_sigma_tot.html')

