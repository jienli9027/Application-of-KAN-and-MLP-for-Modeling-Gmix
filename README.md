# Application of Kolmogorov-Arnold and Deep Neural Networks for Modeling Gibbs Free Energy of Mixing Modified by Global Renormalization Group Theory

Author: Ji-En, Lo (羅翊心)
Institution: Department of Chemical Engineering, National Tsing Hua University
Date: July 2025
# Overview

### Background
- Traditional local composition models are widely used to model phase separation in multicomponent mixtures.
- However, they fail to capture long-range fluctuation effects near the critical point.
- GRGT transforms mean-field theory to include these fluctuations, ensuring that the correlation length obeys nonclassical scaling laws as \( T \rightarrow T_c \).

### Challenge
- GRGT does **not** yield a closed-form expression for the Gibbs free energy of mixing \( g_{mix}(x, T) \).
- This makes it difficult to directly compute phase equilibria using classical methods.

### Solution: Surrogate Modeling via Neural Networks
- Neural networks are used to approximate the GRGT-modified excess Gibbs free energy \( g_{ex}^{RG} \).
- The surrogate models are differentiable and allow accurate computation of thermodynamic derivatives.
- Two neural network architectures are tested:
  - **Multi-Layer Perceptron (MLP)**
  - **Kolmogorov–Arnold Network (KAN)**

### Applications
- The surrogate models are applied to compute liquid–liquid equilibrium using:
  - Two-suffix Margules model
  - Non-Random Two-Liquid (NRTL) model
- Both models are tested for their ability to fit \( g_{ex} \) and its derivatives.

### Key Findings
- MLP yields better predictions of derivatives (e.g., \( \partial g / \partial x \), \( \partial g / \partial T \)) and satisfies Gibbs–Duhem relation more accurately.
- KAN achieves higher accuracy for \( g_{ex} \) in data-scarce scenarios, but with less accurate derivative behavior.
- ReLU-type activation functions are **not suitable** due to zero second derivatives, which conflict with thermodynamic consistency requirements.
