import numpy as np
import yfinance as yf
import scipy as sp
import pandas as pd

def heston_pricer_robust(S0, K, T, r, v0, kappa, theta, rho, sigma):

    K = np.atleast_1d(K)
    num_phi_points = 2000
    phi_max = 50.0
    phi = np.linspace(1e-6, phi_max, num_phi_points)

    # --- P1 and P2 Calculation ---
    P = np.zeros((2, len(K))) # Array to store P1 and P2 results

    for j in [1, 2]:
        if j == 1:
            u, b = 0.5, kappa - rho * sigma
        else: # j == 2
            u, b = -0.5, kappa
        
        a = kappa * theta
        with np.errstate(all='ignore'): # Suppress warnings during optimization
            d = np.sqrt((rho * sigma * 1j * phi - b)**2 + sigma**2 * (phi**2 - 2 * u * 1j * phi))
            g = (b - rho * sigma * 1j * phi - d) / (b - rho * sigma * 1j * phi + d)
            exp_neg_dT = np.exp(-d * T)
            # This is a more stable form of the characteristic function's C and D components
            C = r * 1j * phi * T + (a / sigma**2) * ((b - rho * sigma * 1j * phi - d) * T - 2 * np.log((1 - g * exp_neg_dT) / (1 - g)))
            D = ((b - rho * sigma * 1j * phi - d) / sigma**2) * ((1 - exp_neg_dT) / (1 - g * exp_neg_dT))

        char_func = np.exp(C + D * v0 + 1j * phi * np.log(S0))
        
        # Vectorized integrand calculation
        z_integrand_numerator = np.exp(-1j * phi[np.newaxis, :] * np.log(K[:, np.newaxis])) * char_func[np.newaxis, :]
        integrand = np.imag(z_integrand_numerator) / phi[np.newaxis, :]

        #Graph of heston Integrand for P1 and P2
        """plt.figure(figsize=(10,5))
        plt.plot(phi, integrand[0,:]),
        label = "Heston Integrand for P"
        color ='blue'

        plt.title("Visualisation de la convergence de l'intégrale de Fourier-Heston")
        plt.xlabel("Fréquence(Phi)")
        plt.ylabel("Valeur de l'intégrande")
        plt.axhline(0, color='black', lw=0.5, ls='--')
        plt.grid(True, alpha=0.3)
        plt.show()"""

        # Integration
        integral = sp.integrate.simpson(integrand, x=phi, axis=1)
        P[j-1, :] = 0.5 + (1 / np.pi) * integral

    # Final price calculation
    price = S0 * P[0, :] - K * np.exp(-r * T) * P[1, :]
    
    # Return a single value if only one strike was passed
    if len(price) == 1:
        return price[0]
    return price

def heston_monte_carlo_pricer(S0,K,T,r,v0,kappa,theta, rho,sigma,num_simulations,num_steps):
    dt = T / num_steps  # Pas de temps
    v = np.zeros((num_steps + 1, num_simulations))  # Tableau pour chaque jour de chaque simulation
    v[0] = v0  # Première valeur égal la valeur initiale
    Z1 = np.random.normal(size=(num_steps, num_simulations))  # Première matrice du hasard
    Z2 = np.random.normal(size=(num_steps, num_simulations))  # Deuxième matrice du hasard
    W1 = Z1  # Premier facteur aléatoire
    W2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2  # Deuxième facteur aléatoire corrélé au premier
    for t in range(1, num_steps + 1):  # Trouve la valeur de la volatilité pour chauqe case de la matrice v
        v_prev = np.maximum(v[t - 1, :], 0)  # Évite les valeurs négatives
        # Calcul la valeur actuelle de v (deuxième EDS de Heston)
        v[t, :] = v[t - 1, :] + kappa * (theta - v_prev) * dt + sigma * np.sqrt(v_prev * dt) * W2[t - 1,:]
    v_trunc = np.maximum(v[:-1, :], 0)  # On s'assure encore de ne pas avoir de valeur négative
    log_returns = (r - 0.5 * v_trunc) * dt + np.sqrt(v_trunc * dt) * W1  # Calcul du rendement (première EDS de Heston)
    total_log_returns = np.sum(log_returns, axis=0)  # Somme de toute les rendements
    S_final = S0 * np.exp(total_log_returns)  # Caclul de la matrice des prix finals en fonction de la valeur de départ
    payoffs = np.maximum(S_final - K, 0)
    option_price_mc = np.exp(-r * T) * np.mean(payoffs) # Prend en compte le temps des interêts composés
    return option_price_mc

def parameters(action):
    # Téléchargement des données
    df = yf.download(action, period="10y", interval="1d", progress=False)
    if df.empty: # S'assure d'avoir des données
        raise ValueError(f"Aucune donnée trouvée pour {action}")

    # Gestion des MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # Si le premier niveau est le ticker, on le supprime
        if action in df.columns.levels[0]:
            df.columns = df.columns.droplevel(0)
        else:
            df.columns = df.columns.droplevel(1)

    # Si "Adj Close" n'existe pas, on utilise "Close"
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    
    # Prix initial (S0)
    S0 = df[col].iloc[-1]

    # Calcul du Daily Range (Estimateur de Parkinson) comme proxy de la variance instantanée
    # v_t approx (ln(High/Low)^2) / (4 * ln(2))
    df["Parkinson_Var"] = ((np.log(df["High"] / df["Low"]))**2) / (4 * np.log(2)) * 252
    
    # Rendements logarithmiques quotidiens
    df["Return"] = np.log(df[col] / df[col].shift(1))
    
    # On utilise une version légèrement lissée (5 jours) de Parkinson pour réduire le bruit
    # Cela aide à mieux voir la corrélation rho qui est souvent masquée par le bruit quotidien
    df["Var_5d"] = df["Parkinson_Var"].rolling(5).mean()
    df = df.dropna()

    # Initial variance (v0)
    v0 = df["Var_5d"].iloc[-1]

    # Long-term variance (theta)
    theta = df["Var_5d"].mean()

    # Kappa (Vitesse de retour à la moyenne)
    y_k = df["Var_5d"].values[1:]
    x_k = df["Var_5d"].values[:-1]
    beta, alpha = np.polyfit(x_k, y_k, 1)
    kappa = -np.log(max(beta, 0.0001)) * 252

    # Xi (Vol-of-vol) et Rho (Corrélation)
    dt = 1/252
    v_t = df["Var_5d"].values[:-1]
    v_next = df["Var_5d"].values[1:]
    
    # Innovations de la variance lissée
    target_dv = v_next - v_t - kappa * (theta - v_t) * dt
    variance_innovations = target_dv / np.sqrt(np.maximum(v_t, 1e-6))
    
    xi = np.std(variance_innovations) * np.sqrt(252)

    # Rho (Corrélation)
    # Note: L'estimation historique de rho est souvent plus faible (ex: -0.2) que celle calibrée sur les options (-0.7).
    returns = df["Return"].values[1:]
    rho = np.corrcoef(returns, variance_innovations)[0, 1]

    # Taux sans risque (r)
    try:
        ticker_r = yf.Ticker("CBIL.TO")
        r = ticker_r.info.get("trailingAnnualDividendYield")
        if r is None or r <= 0:
            r = 0.0225
    except:
        r = 0.0225

    return float(S0), float(v0), float(r), float(theta), float(kappa), float(xi), float(rho)
# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':

    # --- Part 1: Theoretical Pricing (Monte Carlo vs Analytical) ---
    print("--- Part 1: Theoretical Pricing ---")
    # Parameters for theoretical pricing
    S0_th = 100 # Prix initial
    K_th = 100  # Strike
    T_th = 1.0 # Temps d'expiration
    r_th = 0.05 # Taux sans risque
    v0_th = 0.04 # Volatilité initial
    kappa_th = 2.0 # Retour vers la moyenne
    theta_th = 0.04 # Impact du temps
    rho_th = -0.7 # corrélation entre S et v
    sigma_th = 0.3 # Volatilité de la volatilité
    num_simulations_th = 400000 # Nombre de simulation
    num_steps_th = 252 # Nombre d'itération (jours) par simulation



    #option_price_mc = heston_monte_carlo_pricer(S0_th,K_th,T_th,r_th,v0_th,kappa_th,theta_th,rho_th,sigma_th,num_simulations_th,num_steps_th)
    #print(f"European Call Option Price (Monte Carlo): {option_price_mc:.4f}")

    # Analytical Price (for comparison)
    #analytical_price_theoretical = heston_pricer_robust(S0_th, K_th, T_th, r_th, v0_th, kappa_th, theta_th, rho_th, sigma_th)
    #print(f"Analytical European Call Option Price (Theoretical): {analytical_price_theoretical:.4f}")
    action = "SPY"
    S0, v0, r, theta, kappa, xi, rho = parameters(action)
    print("Action choisi :", action)
    print("Valeur de S0 :",S0)
    print("Valeur de v0 :", v0)
    print("Valeur de r :", r)
    print("Valeur de theta :", theta)
    print("Valeur de kappa :", kappa)
    print("Valeur de xi :", xi)
    print("Valeur de rho :", rho)
    option_price_mc = heston_monte_carlo_pricer(S0_th,K_th,T_th,r_th,v0_th,kappa_th,theta_th,rho_th,sigma_th,num_simulations_th,num_steps_th)
    print(f"European Call Option Price (Monte Carlo): {option_price_mc:.4f}")
