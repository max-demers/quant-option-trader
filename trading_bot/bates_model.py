import numpy as np
from scipy.optimize import differential_evolution, brentq
from scipy.integrate import simpson
from scipy.stats import norm

class BatesModel:
    """
    Modèle de Bates: calibration,princing et analyse
    """
    def __init__(self):
        """
        Initialise le modèle avec aucun paramètre. Il doit être calibré.
        """
        self.v0 = None
        self.kappa = None
        self.theta = None
        self.rho = None
        self.sigma = None
        self.lambda_jump = None
        self.mu_jump = None
        self.sigma_jump = None
        self.rmse = None

    def bates_pricer_robust(self, S0, K, T, r):
        """
        Prix des options de call utilisant les paramèetre du modèle calibré de Bates.
        """
        if any(p is None for p in [self.v0, self.kappa, self.theta, self.rho, self.sigma, self.lambda_jump, self.mu_jump, self.sigma_jump]):
            raise ValueError("Model parameters must be calibrated before pricing.")

        K = np.atleast_1d(K)
        num_phi_points = 2000
        phi_max = 50.0
        phi = np.linspace(1e-6, phi_max, num_phi_points)

        P = np.zeros((2, len(K)))

        for j in [1, 2]:
            u, b = (0.5, self.kappa - self.rho * self.sigma) if j == 1 else (-0.5, self.kappa)
            a = self.kappa * self.theta
            with np.errstate(all='ignore'):
                d = np.sqrt((self.rho * self.sigma * 1j * phi - b)**2 + self.sigma**2 * (phi**2 - 2 * u * 1j * phi))
                g = (b - self.rho * self.sigma * 1j * phi - d) / (b - self.rho * self.sigma * 1j * phi + d)
                exp_neg_dT = np.exp(-d * T)
                k_jump = self.lambda_jump * (np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1)
                
                C = (r - k_jump) * 1j * phi * T + (a / self.sigma**2) * \
                    ((b - self.rho * self.sigma * 1j * phi - d) * T - 2 * np.log((1 - g * exp_neg_dT) / (1 - g)))
                
                D = ((b - self.rho * self.sigma * 1j * phi - d) / self.sigma**2) * ((1 - exp_neg_dT) / (1 - g * exp_neg_dT))
                
                jump_exponent = T * self.lambda_jump * (np.exp(1j * phi * self.mu_jump - 0.5 * phi**2 * self.sigma_jump**2) - 1)
                
                char_func = np.exp(C + D * self.v0 + 1j * phi * np.log(S0) + jump_exponent)
            
            z_integrand_numerator = np.exp(-1j * phi[np.newaxis, :] * np.log(K[:, np.newaxis])) * char_func[np.newaxis, :]
            integrand = np.imag(z_integrand_numerator) / phi[np.newaxis, :]
            
            integral = simpson(integrand, x=phi, axis=1)
            P[j - 1, :] = 0.5 + (1 / np.pi) * integral

        price = S0 * P[0, :] - K * np.exp(-r * T) * P[1, :]
        delta = P[0, :]

        return price, delta

    @staticmethod
    def _objective_function(params, market_data, S0, r):
        """
        Fonction statique pour l'optimiseur.
        """
        v0, kappa, theta, rho, sigma, lambda_jump, mu_jump, sigma_jump = params
        
        # Condition de Feller, retour une grande erreur si pas respecté.
        if 2 * kappa * theta < sigma**2:
            return 1e12

        # Crée une instance de modèle temporaire pour utiliser le pricer.
        temp_model = BatesModel()
        temp_model.v0, temp_model.kappa, temp_model.theta, temp_model.rho, temp_model.sigma, \
        temp_model.lambda_jump, temp_model.mu_jump, temp_model.sigma_jump = params

        all_model_prices = []
        try:
            for T, group in market_data.groupby('T'):
                strikes = group['strike'].values
                model_prices_T, _ = temp_model.bates_pricer_robust(S0, strikes, T, r)
                model_prices_T = np.atleast_1d(model_prices_T)
                if np.isnan(model_prices_T).any():
                    return 1e10
                all_model_prices.append(model_prices_T)

            model_prices = np.concatenate(all_model_prices)
            market_prices = market_data['price'].values
            error = np.sum((model_prices - market_prices)**2)
            
            if np.isnan(error): return 1e10
        except Exception:
            return 1e10
            
        return error

    def calibrate(self, market_data, S0, r, bounds, optimizer_settings):
        """
        Calibrer le modèle pour qu'il réflete la réalité du marché.
        """
        print("\nRunning optimization with differential_evolution for Bates Model...")
        result = differential_evolution(
            self._objective_function,
            bounds,
            args=(market_data, S0, r),
            **optimizer_settings
        )

        if result.success:
            self.v0, self.kappa, self.theta, self.rho, self.sigma, \
            self.lambda_jump, self.mu_jump, self.sigma_jump = result.x
            self.rmse = np.sqrt(result.fun / len(market_data))
            
            print("\nBates Model Calibration Successful!")
            print("\n--- Calibrated Heston Parameters ---")
            print(f"v0={self.v0:.4f}, kappa={self.kappa:.4f}, theta={self.theta:.4f}, rho={self.rho:.4f}, sigma={self.sigma:.4f}")
            print("\n--- Calibrated Jump Parameters ---")
            print(f"lambda={self.lambda_jump:.4f}, mu_jump={self.mu_jump:.4f}, sigma_jump={self.sigma_jump:.4f}")
            print(f"\nFinal Minimized Error (Sum of Squares): {result.fun:.4f}")
            print(f"Root Mean Squared Error (RMSE): ${self.rmse:.2f}")
        else:
            print("\nCalibration failed.")
            print(f"Reason: {result.message}")

    @staticmethod
    def black_scholes_call(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(0, S - K)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def implied_volatility(self, market_price, S, K, T, r):
        if T <= 0: return 0.0
        try:
            objective = lambda sigma: self.black_scholes_call(S, K, T, r, sigma) - market_price
            return brentq(objective, a=1e-3, b=2.0, xtol=1e-6)
        except (ValueError, RuntimeError):
            return np.nan
