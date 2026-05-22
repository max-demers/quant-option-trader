import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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

def heston_finite_differences(S0, K, T, r, kappa, theta, rho, sigma, N_x, N_v, N_t):
    # Grid Setup
    vol_approx = np.sqrt(max(theta, 0.04))
    x_min = np.log(S0) - 5 * vol_approx * np.sqrt(T)
    x_max = np.log(S0) + 5 * vol_approx * np.sqrt(T)
    v_min = 0.0
    v_max = max(4 * theta, 1.0)

    x_axe = np.linspace(x_min, x_max, N_x)
    v_axe = np.linspace(v_min, v_max, N_v)
    dx = x_axe[1] - x_axe[0]
    dv = v_axe[1] - v_axe[0]
    dt = T / N_t

    x_grille, v_grille = np.meshgrid(x_axe, v_axe, indexing="ij")
    v_vec = v_grille.flatten()
    U_plan = np.maximum(np.exp(x_grille) - K, 0.0)
    U = U_plan.flatten()

    # A1: x-derivative operator (depends on v)
    alpha_x_v = -(r - 0.5 * v_vec) / (2 * dx) + (0.5 * v_vec) / (dx**2)
    beta_x_v = -v_vec / (dx**2) - 0.5 * r
    gamma_x_v = (r - 0.5 * v_vec) / (2 * dx) + (0.5 * v_vec) / (dx**2)

    A1 = sp.sparse.diags([alpha_x_v[N_v:], beta_x_v, gamma_x_v[:-N_v]], [-N_v, 0, N_v], shape=(N_x * N_v, N_x * N_v), format="lil")
    # Boundary conditions for x (i=0 and i=N_x-1)
    for j in range(N_v):
        idx_0 = j
        idx_end = (N_x - 1) * N_v + j
        A1[idx_0, :] = 0; A1[idx_0, idx_0] = -0.5 * r
        A1[idx_end, :] = 0; A1[idx_end, idx_end] = -0.5 * r
    A1 = A1.tocsr()


    # A2: v-derivative operator
    alpha_v = -kappa * (theta - v_axe) / (2 * dv) + (0.5 * sigma**2 * v_axe) / (dv**2)
    beta_v = -(sigma**2 * v_axe) / (dv**2) - 0.5 * r
    gamma_v = kappa * (theta - v_axe) / (2 * dv) + (0.5 * sigma**2 * v_axe) / (dv**2)

    A2_v = sp.sparse.diags([alpha_v[1:], beta_v, gamma_v[:-1]], [-1, 0, 1], shape=(N_v, N_v), format="lil")
    A2_v[0, :] = 0; A2_v[0, 0] = -0.5 * r
    A2_v[-1, :] = 0; A2_v[-1, -1] = -0.5 * r
    A2 = sp.sparse.kron(sp.sparse.eye(N_x), A2_v.tocsr(), format="csr")

    # A0: Mixed derivative operator rho * sigma * v * U_xv
    A0 = sp.sparse.lil_matrix((N_x * N_v, N_x * N_v))
    coef_mixte = 0.25 * rho * sigma * v_vec / (dx * dv)
    for i in range(1, N_x - 1):
        for j in range(1, N_v - 1):
            idx = i * N_v + j
            c = coef_mixte[idx]
            A0[idx, (i + 1) * N_v + (j + 1)] = c
            A0[idx, (i + 1) * N_v + (j - 1)] = -c
            A0[idx, (i - 1) * N_v + (j + 1)] = -c
            A0[idx, (i - 1) * N_v + (j - 1)] = c
    A0 = A0.tocsr()

    A_total = A0 + A1 + A2
    theta_param = 0.5 + np.sqrt(6)/6
    I = sp.sparse.eye(N_x * N_v, format="csr")
    LHS_A1 = sp.sparse.linalg.factorized((I - theta_param * dt * A1).tocsc())
    LHS_A2 = sp.sparse.linalg.factorized((I - theta_param * dt * A2).tocsc())

    for n in reversed(range(N_t)):
        # Modified Craig-Sneyd (MCS) Scheme
        # Prediction
        Y0 = U + dt * (A_total @ U)
        Y1 = LHS_A1(Y0 - theta_param * dt * (A1 @ U))
        Y2 = LHS_A2(Y1 - theta_param * dt * (A2 @ U))
        
        # Correction
        Z0 = Y0 + 0.5 * dt * (A0 @ (Y2 - U)) + (theta_param - 0.5) * dt * (A1 @ (Y1 - U) + A2 @ (Y2 - U))
        Y3 = LHS_A1(Z0 - theta_param * dt * (A1 @ Y2))
        U = LHS_A2(Y3 - theta_param * dt * (A2 @ Y2))

        # Enforce Boundary Conditions
        U_mat = U.reshape((N_x, N_v))
        t_actuel = n * dt
        U_mat[-1, :] = np.exp(x_max) - K * np.exp(-r * (T - t_actuel))
        U_mat[0, :] = 0.0
        U = U_mat.flatten()

    U_final = U.reshape((N_x, N_v))
    return x_axe, v_axe, U_final



# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':

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
    num_simulations_th = 500000 # Nombre de simulation
    num_steps_th = 252 # Nombre d'itération (jours) par simulation
    N_t = 300
    N_x = 800
    N_v = 400

    action = "SPY"
    print("Action choisi :", action)
    print("Valeur de S0 :",S0_th)
    print("Valeur de v0 :", v0_th)
    print("Valeur de r :", r_th)
    print("Valeur de theta :", theta_th)
    print("Valeur de kappa :", kappa_th)
    print("Valeur de vol-of-vol :", sigma_th)
    print("Valeur de rho :", rho_th)

    print("Calcul du prix de l'option en cours...")

    x_axe, v_axe, grille_prix = heston_finite_differences(S0_th,K_th,T_th,r_th,v0_th,kappa_th,theta_th,rho_th,sigma_th,N_x, N_v,N_t)
    interp = sp.interpolate.RegularGridInterpolator((x_axe, v_axe), grille_prix, method="linear")
    prix_heston_adi = interp([np.log(S0_th), v0_th])[0]
    print(f"European Call option Price (ADI): {prix_heston_adi: .4f}")


    option_price_mc = heston_monte_carlo_pricer(S0_th,K_th,T_th,r_th,v0_th,kappa_th,theta_th,rho_th,sigma_th,num_simulations_th,num_steps_th)
    print(f"European Call Option Price (Monte Carlo): {option_price_mc:.4f}")
    

    analytical_price_theoretical = heston_pricer_robust(S0_th, K_th, T_th, r_th, v0_th, kappa_th, theta_th, rho_th, sigma_th)
    print(f"Analytical European Call Option Price (Theoretical): {analytical_price_theoretical:.4f}")

