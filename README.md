 A research project with Imperial College London on options with stochastic interest volatility.
 
``` python
import random
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
import matplotlib.pyplot as plt
from functools import cached_property

rng = np.random.default_rng(12345)
```

# Defining Options

Analytical formulae for calls & puts:

$$\begin{align*}
C_0 & = \mathbb{E}[e^{-rT}(S_T-K)^+] \\
    & = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2) \\\\    
P_0 & = \mathbb{E}[e^{-rT}(K-S_T)^+] \\
    & = Ke^{-rT}\Phi(-d_2)-S_0\Phi(-d_1)
\end{align*}$$

Where:

$$\begin{align*}
d_1 & = \frac{\log(\frac{S_0}{K}) + (r + 0.5 \sigma^2)T }{\sigma \sqrt{T}}, \\
d_2 & = d_1 - \sigma \sqrt{T}.  
\end{align*}$$

``` python
def calculate_d1_d2(s0, k, T, r, sigma):
  d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * T)/(sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return d1, d2

class Option:
  def __init__(self, name, price, payoff):
    self.name = name
    self.price = price
    self.payoff = payoff

def call_price(s0, k, T, r, sigma):
  d1, d2 = calculate_d1_d2(s0, k, T, r, sigma)
  return s0 * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)
def call_payoff(sT, k):
  return np.maximum(sT - k, 0)

def put_price(s0, k, T, r, sigma):
  d1, d2 = calculate_d1_d2(s0, k, T, r, sigma)
  return k * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1)
def put_payoff(sT, k):
  return np.maximum(k - sT, 0)
```

# Option Portfolio

``` python
class Portfolio:
    def __init__(self, *positions):
      self.positions = positions

    def describe(self):
      """Print strategy details."""
      for option, strike, qty in self.positions:
          action = "Buy" if qty > 0 else "Sell"
          print(f"  {action} {abs(qty)} {option.name} at K={strike}")

    def payoff(self, sT : np.ndarray):
      return sum(qty * option.payoff(sT, strike) for option, strike, qty in self.positions)

    def analytical_price(self, s0, t, r, sigma):
      return sum(qty * option.price(s0, strike, t, r, sigma) for option, strike, qty in self.positions)

    def get_payoff_graph(self, stock_range=(50, 150), ax = None):
      """Plots the payoff diagram for the strategy."""
      if ax is None:
        fig, ax = plt.subplots()
      else:
        fig = ax.figure

      stock_prices = np.linspace(stock_range[0], stock_range[1], 200)
      payoffs = self.payoff(stock_prices)

      ax.plot(stock_prices, payoffs, label="Strategy Payoff", color='b', linewidth=2)

      # Breakeven line
      ax.axhline(0, color='black', linewidth=1, linestyle='--')

      # First strike price
      ax.axvline(x=self.positions[0][1], color='gray', linestyle='--', alpha=0.6)

      ax.set_xlabel("Stock Price at Expiration")
      ax.set_ylabel("Profit / Loss")
      ax.set_title("Option Strategy Payoff Diagram")
      ax.grid(True)
      return fig

    def graph_payoff(self):
      fig = self.get_payoff_graph()
      fig.show()
```

Example option strategies:

``` python
Call = Option("Call", call_price, call_payoff)
Put = Option("Put", put_price, put_payoff)

bull_call_spread = Portfolio(
    (Call, 100, +1),
    (Call, 110, -1)
)

iron_condor = Portfolio(
    (Put, 90, +1),
    (Put, 95, -1),
    (Call, 105, -1),
    (Call, 110, +1)
)

straddle = Portfolio(
    (Call, 100, +1),
    (Put, 100, +1)
)

# bull_call_spread.graph_payoff()
# iron_condor.graph_payoff()
# straddle.graph_payoff()
```

# Asset Simulator

Terminal price of an asset:

$$\begin{align*}
S_T & = S_0 e^{(r - 0.5 \sigma^2)T + \sigma W_T}
\end{align*}$$

``` python
class AssetSimulator():
    def __init__(self, rng, s0, t, r, sigma, simulation_n):
        """Simulates terminal asset price n times"""
        self.rng = rng
        self.s0 = s0
        self.t = t
        self.r = r
        self.sigma = sigma
        self.simulation_n = simulation_n

    @property
    def discount(self):
        """Discount under constant interest r"""
        return np.exp(-self.r * self.t)

    @cached_property
    def sT(self) -> np.ndarray:
        """Terminal stockprice sT"""
        tmp1 = (self.r - 0.5*(self.sigma ** 2)) * self.t
        tmp2 = self.sigma * np.sqrt(self.t) * self.rng.standard_normal(size=self.simulation_n)
        return self.s0 * np.exp(tmp1 + tmp2)

    @cached_property
    def discounted_sT(self) -> np.ndarray:
        return np.exp(-self.r * self.t) * self.sT

    def get_hist(self, ax = None):
      """Creates histogram of sT"""
      if ax is None:
        fig, ax = plt.subplots()
      else:
        fig = ax.figure

      ax.hist(self.sT, bins=100)
      ax.set_ylabel("Frequency")
      ax.set_xlabel("$S_T$")
      ax.set_title("Histogram of simulated $S_T$")
      return fig

    def display_hist(self):
      fig = self.get_hist()
      fig.show()
```

# Option Simulator

Value at Risk & Expected Shortfall:

$$
\mathbb{P}\!\big[ L_H < \text{VaR}_{H,\alpha} \big] = \alpha
$$

$$
\begin{aligned}
\text{ES}_{H,\alpha} &= \mathbb{E}^{\mathbb{P}}\!\big[ L_H \mid L_H > \text{VaR}_{H,\alpha} \big] \\
                     &= \frac{ \mathbb{E}^{\mathbb{P}}\!\big[ L_H \, 1_{\{ L_H > \text{VaR}_{H,\alpha} \}} \big] }{ 1 - \alpha }
\end{aligned}
$$

Where:

$$L_H = \text{Portfolio}_0 - \text{Portfolio}_H$$

To estimate $\mathbb{E}[X]$, we can use the control variate estimator.  
Given a $Y$ such that $\mathbb{E}[Y]$ is known:

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^{n} \big( X_i + c^{*}\big( Y_i - \mathbb{E}[Y] \big) \big),
\qquad
c^{*} = -\frac{\operatorname{cov}(X,Y)}{\operatorname{var}(Y)} .
$$

``` python
class OptionSimulator():
    def __init__(self, asset : AssetSimulator, portfolio : Portfolio):
      """Calculates various properties of a portfolio"""
      self.asset = asset
      self.portfolio = portfolio

    @cached_property
    def discounted_payoff(self) -> np.ndarray:
      return self.asset.discount * self.portfolio.payoff(self.asset.sT)

    @cached_property
    def z(self) -> np.ndarray:
      """Control variate simulation: each z_i is an estimator for the option price"""
      cstar = -np.cov(self.discounted_payoff,
                      self.asset.discounted_sT,
                      ddof=1)[0, 1] / np.var(self.asset.discounted_sT, ddof=1)
      z = self.discounted_payoff + cstar * (self.asset.discounted_sT - self.asset.s0)
      return z

    @cached_property
    def eZ(self) -> float:
      """E[z]"""
      return np.mean(self.z)

    @cached_property
    def varZ(self) -> float:
      """Var[z]"""
      return np.var(self.z)

    @cached_property
    def LH(self) -> np.ndarray:
      """Loss over time horizon: difference between portfolio value at time 0 and time T"""
      portfolio_0 = self.portfolio.analytical_price(
          self.asset.s0, self.asset.t, self.asset.r, self.asset.sigma
      )
      return portfolio_0 - self.discounted_payoff

    @property
    def rhosquared(self) -> float:
      """Squared correlation between discounted payoff and discounted stock"""
      cov = np.cov(self.asset.discounted_sT, self.discounted_payoff, ddof=1)[0, 1]
      varx = np.var(self.discounted_payoff, ddof=1)
      vary = np.var(self.asset.discounted_sT, ddof=1)
      rhosquared = (cov ** 2) / (varx * vary)
      return rhosquared

    def CI(self, epsilon = 0.05) -> tuple:
      """Confidence interval with SL epsilon"""
      sigma_est = self.eZ / np.sqrt(self.asset.simulation_n)
      aepsilon = norm.ppf(1.0 - epsilon * 0.5)

      # Left and right boundaries of CI
      ci_left = self.eZ - aepsilon * sigma_est
      ci_right = self.eZ + aepsilon * sigma_est

      return ci_left, ci_right

    def VaR(self, alpha = .95) -> float:
      # LH is negative so percentile is 1 - alpha
      return np.percentile(self.LH, (1 - alpha) * 100)

    def ES(self, alpha = .95) -> float:
      # LH is negative so condition is LH <= VaR
      return np.mean(self.LH * (self.LH <= self.VaR(alpha))) / (1 - alpha)

    def summarise(self, dp: int = 5):
      """Prints details of the simulation."""
      try:
        const_rate = True
        analytical = self.portfolio.analytical_price(
            self.asset.s0, self.asset.t, self.asset.r, self.asset.sigma
        )
      except AttributeError:
        const_rate = False
        analytical = None

      mc_price = self.eZ
      mc_variance = self.varZ
      n_paths = self.asset.simulation_n
      std_error = np.sqrt(mc_variance) / np.sqrt(n_paths)
      crit = norm.ppf(1.0 - 0.05 / 2.0)
      ci_left = mc_price - crit * std_error
      ci_right = mc_price + crit * std_error

      portfolio_0 = analytical if const_rate else mc_price
      rho_sq = self.rhosquared

      print("\n=== OPTION‑PORTFOLIO SIMULATION SUMMARY ===")
      if const_rate:
          print(f"Analytical (BS) price : {round(analytical, dp)}")
      else:
          print("Analytical price       : N/A (stochastic short‑rate)")
      print(f"Monte‑Carlo price      : {round(mc_price, dp)}")
      print(f"Monte‑Carlo variance   : {round(mc_variance, dp)}")
      print(f"95 % CI (price)        : ({ci_left:.{dp}f}, {ci_right:.{dp}f})")
      print(f"VaR (95 %)             : {round(self.VaR(), dp)}")
      print(f"ES  (95 %)             : {round(self.ES(), dp)}")
      print(f"ρ² (payoff vs. S_T)   : {round(rho_sq, dp)}")

    def get_pnl_hist(self, alpha = 0.95, ax = None):
        """Returns a histogram figure of P&L with VaR and ES marked."""
        if ax is None:
          fig, ax = plt.subplots()
        else:
          fig = ax.figure

        ax.hist(self.LH, bins=50, alpha=0.75, color='blue')
        ax.set_xlabel("$L_H$")
        ax.set_title("Histogram of $L_H$")

        # VaR and ES lines
        var_value = self.VaR(alpha)
        es_value = self.ES(alpha)
        mean_value = np.mean(self.LH)
        ax.axvline(x=var_value, color='r', linestyle='--', label=f'VaR ({var_value:.2f})')
        ax.axvline(x=es_value, color='b', linestyle='--', label=f'ES ({es_value:.2f})')
        ax.axvline(x=mean_value, color='y', linestyle='--', label=f'Mean ({mean_value:.2f})')

        ax.legend()

        return fig

    def display_pnl(self):
      fig = self.get_pnl_hist()
      plt.show()

    def display_graphs(self):
      """Displays 3 Graphs: Portfolio, Simulated Stocks and P&L Graphs"""
      fig, axes = plt.subplots(1, 3, figsize=(15, 4))
      self.portfolio.get_payoff_graph(ax=axes[0])
      self.asset.get_hist(ax=axes[1])
      self.get_pnl_hist(ax=axes[2])
      fig.show()
```

# Single Simulation

``` python
S0 = 100.0
T = 0.25
R = 0.05
Sigma = 0.3
Samplesize = 10000
Epsilon = 0.05
```

Plot asset\'s terminal stock price.

Compare analytical results with monte-carlo results.

Plot loss over time horizon of the strategy.

``` python
assetSimulator = AssetSimulator(rng, S0, T, R, Sigma, Samplesize)

sim = OptionSimulator(assetSimulator, straddle)
sim.display_graphs()
sim.summarise()
```


    === OPTION‑PORTFOLIO SIMULATION SUMMARY ===
    Analytical (BS) price : 11.92395
    Monte‑Carlo price      : 11.89991
    Monte‑Carlo variance   : 81.75312
    95 % CI (price)        : (11.72269, 12.07712)
    VaR (95 %)             : -17.85086
    ES  (95 %)             : -25.21919
    ρ² (payoff vs. S_T)   : 0.08504

![](single-summary.png)

# Simulation With Interest Rates

OptionSimulator can monitor how VaR and ES shifts with interest rate.

``` python
def generateAssetSimulations(lower, upper, n):
  return [AssetSimulator(rng, S0, T, r, Sigma, Samplesize)
          for r in np.linspace(lower, upper, n)]

def generateOptionSimulations(assetSimulations, option):
  return [OptionSimulator(asset, option) for asset in assetSimulations]

n = 5
asset_sims = generateAssetSimulations(0.01, 0.1, n)
option_sims = generateOptionSimulations(asset_sims, straddle)

fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
for i, sim in enumerate(option_sims):
  sim.get_pnl_hist(ax=axes[i])
  axes[i].set_title(f'r = {sim.asset.r:.2f}')
fig.show()
```

![](r-sims.png)

# Vasicek Model

Perhaps more interestingly, we can observe how these risk measures may vary with respect to other interest-related parameters, like volatility in interest. Instead of modelling $r$ as a constant, we can model short term interest rates stochastically using the Vasicek model.

Under the Vasicek model:

$$
\begin{align*}
dr_t & = k(\theta-r_t)dt+\sigma_2 dW^{(1)}_t \\
dS_t & = r_tS_tdt+\sigma_1S_tdW^{(2)}_t
\end{align*}
$$

To construct two correlated weiner samples $dW^{(1)}_t$ and $dW^{(2)}_t$, we apply a Cholesky decomposition to the covariance matrix to yield the transformation

$$
\begin{align*}
dW^{(1)}_t &= dW_{t,1} \\
dW^{(2)}_t &= \rho \cdot dW_{t,1} + \sqrt{1 - \rho^2} \cdot dW_{t,2}
\end{align*}
$$

## Naïve Solution

We assume $r_t$ is constant over the interval $[t, t+dt]$. One could write

$$S_{t+dt}=S_t+r_nS_ndt+\sigma_1S_tdW^{(2)}_t$$

But, this has conditional expectation $\mathbb{E}[S_{t+dt}|\mathcal{F}_t]=S_t(1+r_tdt)$ (because $dW_t^{(2)}$ has zero mean) and drifts linearly upward. It is no longer a martingale process and exhibits a distorted large-tail log-normal distribution for $S_t$. We should instead keep the conditional expectation $\mathbb{E}[S_{t+dt}|\mathcal{F}_t]=S_te^{r_tdt}$.

## Exact Solution

Applying Itô’s lemma to the log-price $X_t:=\ln S_t$:

$$dX_t = (r_t-\frac{1}{2}\sigma^2_1)dt+\sigma_1dW_t^{(2)}$$

Similarly, we assume $r_t$ is constant over the interval $[t, t+dt]$. Upon exponentiating, we have

$$S_{t+dt}=S_te^{(r_t-\frac{1}{2}\sigma^2_1)dt+\sigma_1dW_t^{(2)}}$$

which is a better solution for $S_{t+dt}$.

## Interest Distribution

[We can solve for $r_t$ explicitly](https://onlinelibrary.wiley.com/doi/10.1155/2018/7056734) since it is an Ornstein-Uhlenbeck process:

$$
\begin{align*}
\mu & = r_0 e^{-kt} + \theta (1 - e^{-kt}) \\
\sigma & = \sqrt{\frac{\sigma^2}{2k} (1 - e^{-2kt})}
\end{align*}$$

``` python
class InterestSimulator:
    """Vasicek model."""
    def __init__(self, rng, r0, t, k, theta, sigma, simulation_n, dt):
        self.rng = rng
        self.r0 = r0
        self.t = t
        self.k = k
        self.theta = theta
        self.sigma = sigma
        self.simulation_n = simulation_n
        self.dt = dt

    @property
    def steps(self) -> int:
        return int(np.floor(self.t / self.dt))

    @cached_property
    def norm_samples(self) -> np.ndarray:
        return self.rng.standard_normal((self.simulation_n, self.steps - 1))

    @cached_property
    def rT(self) -> np.ndarray:
        steps = self.steps
        r = np.full((self.simulation_n, steps), self.r0, dtype=float)

        for s in range(1, steps):
            dr = (self.k * (self.theta - r[:, s - 1]) * self.dt
                  + self.sigma * self.norm_samples[:, s - 1])
            r[:, s] = r[:, s - 1] + dr
        return r

    def rT_sol(self, t):
        """Analytical mean & std of r_t."""
        mu = self.r0 * np.exp(-self.k * t) + self.theta * (1 - np.exp(-self.k * t))
        var = (self.sigma ** 2) / (2 * self.k) * (1 - np.exp(-2 * self.k * t))
        return mu, np.sqrt(var)

    def get_r_graph(self, alpha=0.1, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        t_vals = np.linspace(0, self.t, self.steps)
        ax.plot(t_vals, self.rT.T, color="steelblue", alpha=alpha, lw=0.7)

        mu, sigma = self.rT_sol(t_vals)
        ax.plot(t_vals, mu, color="red", label="Theoretical mean")
        ax.fill_between(
            t_vals,
            mu - 2 * sigma,
            mu + 2 * sigma,
            color="red",
            alpha=0.2,
            label="95 % CI",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$r_t$")
        ax.set_title("Vasicek Simulations")
        ax.legend()

        return fig
```

``` python
class AssetInterestSimulator(AssetSimulator):
    def __init__(self, interest, rho, s0, sigma):
        self.interest = interest
        self.rho = rho
        super().__init__(
            rng=interest.rng,
            s0=s0,
            t=interest.t,
            r=interest.r0,
            sigma=sigma,
            simulation_n=interest.simulation_n
        )

    @cached_property
    def sT(self) -> np.ndarray:
        st = np.full((self.simulation_n, self.interest.steps), self.s0, dtype=float)
        for s in range(1, self.interest.steps):
            dW_r = self.interest.norm_samples[:, s - 1]
            dW_ind = self.rng.standard_normal(self.simulation_n)
            dW = self.rho * dW_r + np.sqrt(1 - self.rho ** 2) * dW_ind

            # GBM with stochastic drift r_t
            drift = (self.interest.rT[:, s - 1] - 0.5 * self.sigma ** 2) * self.interest.dt
            diffusion = self.sigma * np.sqrt(self.interest.dt) * dW

            st[:, s] = st[:, s - 1] * np.exp(drift + diffusion)

        return st[:, -1]

    @cached_property
    def discounted_sT(self) -> np.ndarray:
        """Discounts the asset price using the integrated interest rate."""
        # Approximate the integral of r_t over [0, T] using simpson rule
        integral_r = integrate.simpson(self.interest.rT, dx=self.interest.dt, axis=1)
        discount_factor = np.exp(-integral_r)
        return discount_factor[:, np.newaxis] * self.sT
```

# Single Simulation

``` python
S0 = 100.0
T = 0.25
R = 0.05
Sigma = 0.3
Samplesize = 10000

Theta = 0.02
Rho = 0.5
InterestSigma = 0.05
K = 0.1

interestSimulator = InterestSimulator(rng, R, T, K, Theta, InterestSigma, Samplesize, 0.01)
assetSimulator = AssetInterestSimulator(interestSimulator, Rho, S0, Sigma)

assetSimulator.interest.get_r_graph()

sim = OptionSimulator(assetSimulator, straddle)
sim.display_graphs()
sim.summarise()
```

    === OPTION‑PORTFOLIO SIMULATION SUMMARY ===
    Analytical (BS) price : 11.92395
    Monte‑Carlo price      : 13.02284
    Monte‑Carlo variance   : 94.7415
    95 % CI (price)        : (12.83207, 13.21362)
    VaR (95 %)             : -20.34886
    ES  (95 %)             : -28.86625
    ρ² (payoff vs. S_T)   : 2.76965

![](vasicek-sims.png)

![](vasicek-summary.png)
