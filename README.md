# Risk Modeling Framework

A **Python-based framework for risk modeling** that integrates  
- **Machine Learning algorithms** – in particular **Restricted Boltzmann Machines (RBMs)** for learning probability distributions of borrower and market data  
- **Monte Carlo simulations** for scenario testing and stress analysis  
- **Classical statistical methods** for uncertainty quantification and loss estimation.

The framework is designed to **model the risk of lending decisions**, estimate default probabilities, and quantify potential portfolio losses under a range of macroeconomic scenarios.  
It aims to be **modular, reproducible, and easy to extend** for different credit risk applications.

---

## ✨ Features

- **Probability distribution learning** with RBMs  
- **Monte Carlo engine** for scenario simulation and tail-risk estimation  
- **Model evaluation** with standard risk metrics (e.g., Value at Risk, Expected Shortfall)  
- **Visualization tools** for portfolio risk exposure and scenario comparisons  
- Clear, documented Jupyter notebooks for experimentation and demonstration

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `torch` (for RBM implementation)

```bash
pip install -r requirements.txt
