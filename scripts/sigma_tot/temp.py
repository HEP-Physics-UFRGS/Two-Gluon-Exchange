import numpy as np
from scipy.integrate import fixed_quad
import plotly.graph_objects as go

# === Configurações globais ===
start_sqrt_s = 100
max_sqrt_s = 13000
step = 100
n_points = 10000

b_0 = (33 - 6) / (12 * np.pi)
Lambda = 0.284  # GeV
gamma_1 = 0.084
gamma_2 = 2.36
rho = 4.0
s0 = 1.0  # GeV^2

# --- Escolha do modelo (um mass_model e um ensemble) ---
mass_model = "log"   # pode ser "log" ou "pl"
ensemble = "atlas"   # pode ser "atlas" ou "totem"

# --- Parâmetros do modelo (apenas os necessários) ---
model_params = {
    'atlas': {
        'log': {'mg': 0.334, 'epsilon': 0.061, 'a1': 1.604, 'a2': 3.044},
        'pl':  {'mg': 0.389, 'epsilon': 0.061, 'a1': 1.495, 'a2': 2.161}
    },
    'totem': {
        'log': {'mg': 0.363, 'epsilon': 0.079, 'a1': 1.63, 'a2': 3.28},
        'pl':  {'mg': 0.424, 'epsilon': 0.0775, 'a1': 1.454, 'a2': 2.93}
    }
}

params = model_params[ensemble][mass_model]
mg, epsilon, a1, a2 = params['mg'], params['epsilon'], params['a1'], params['a2']

# === Funções auxiliares ===
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
    return alpha_D_plus * alpha_D_minus * G_p(q2, a1, a2) ** 2

def T_2(k, q, phi, mg, a1, a2, m2_func):
    q2 = q ** 2
    qk_cos = q * k * np.cos(phi)
    qk_plus_squared = q2 / 4 + qk_cos + k ** 2
    qk_minus_squared = q2 / 4 - qk_cos + k ** 2
    alpha_D_plus = alpha_D(qk_plus_squared, mg, m2_func)
    alpha_D_minus = alpha_D(qk_minus_squared, mg, m2_func)
    factor = q2 + 9 * abs(k ** 2 - q2 / 4)
    return alpha_D_plus * alpha_D_minus * G_p(factor, a1, a2) * (2 * G_p(q2, a1, a2) - G_p(factor, a1, a2))

def integrand(y, x, mg, a1, a2, m2_func, sqrt_s):
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

# === Main ===
def main():
    sqrt_s_values = []
    sigma_values = []
    imag_amp_values = []

    m2_func = get_m2_function(mass_model)

    sqrt_s = start_sqrt_s
    while sqrt_s <= max_sqrt_s:
        def inner_integral(x):
            return fixed_quad(
                lambda y: integrand(y, x, mg, a1, a2, m2_func, sqrt_s),
                0, 1, n=n_points
            )[0]

        integral_value = fixed_quad(inner_integral, 0, 1, n=n_points)[0]

        s = sqrt_s ** 2
        amp_value = amp_calculation(integral_value, s, epsilon)
        sigma_value = sigma_tot(amp_value, s)

        sqrt_s_values.append(sqrt_s)
        sigma_values.append(sigma_value)
        imag_amp_values.append(amp_value.imag)

        sqrt_s += step

    # --- Plot 1: Sigma Tot ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=sqrt_s_values, y=sigma_values,
        mode="lines+markers", name=f"{mass_model} {ensemble}",
        line=dict(color="blue"), marker=dict(size=3)
    ))
    fig1.update_layout(
        title=f"Sigma Tot vs. sqrt(s) ({mass_model}, {ensemble})",
        xaxis=dict(title="sqrt(s) [GeV]", type="log"),
        yaxis=dict(title="Sigma Tot [mb]"),
        plot_bgcolor="white"
    )
    fig1.update_xaxes(gridcolor="lightgray")
    fig1.update_yaxes(gridcolor="lightgray")
    fig1.show(renderer="browser")

    # --- Plot 2: Im(Amplitude) ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sqrt_s_values, y=imag_amp_values,
        mode="lines+markers", name=f"Im(Amp) {mass_model} {ensemble}",
        line=dict(color="red"), marker=dict(size=3)
    ))
    fig2.update_layout(
        title=f"Im(Amplitude) vs. sqrt(s) ({mass_model}, {ensemble})",
        xaxis=dict(title="sqrt(s) [GeV]", type="log"),
        yaxis=dict(title="Im(Amp)"),
        plot_bgcolor="white"
    )
    fig2.update_xaxes(gridcolor="lightgray")
    fig2.update_yaxes(gridcolor="lightgray")
    fig2.show(renderer="browser")

if __name__ == "__main__":
    main()
