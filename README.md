# Study of Total and Differential Cross Sections and Beta Parameter Based on Nonperturbative Gluon Exchange in $$pp$$ Elastic Scattering

This repository contains research and analysis related to the paper titled **"Nonperturbative gluon exchange in $$pp$$ elastic scattering at TeV energies"** by G. B. Bopsin et al. The study focuses on the two-gluon-exchange model of the Pomeron, incorporating nonperturbative gluon propagators with dynamical mass scales to describe high-energy proton-proton ($$pp$$) scattering data.

## Key Findings

1. **Total and Differential Cross Sections**:
   - The model successfully describes LHC data for $$pp$$ elastic scattering at $$\sqrt{s} = 7$$, 8, and 13 TeV.
   - The differential cross section $$d\sigma/dt$$ is analyzed using a cumulant expansion for the proton form factor, with the best fit achieved for $$N_a = 2$$ (quadratic in $$|t|$$).
   - The predicted total cross section $$\sigma_{\text{tot}}(s = 13\,\text{TeV})$$ for each configuration is:
     - Ensemble A, $$m_{\text{log}}(q^2)$$: $$\sigma_{\text{tot}} = 104.3\,\text{mb}$$  
     - Ensemble A, $$m_{\text{pl}}(q^2)$$: $$\sigma_{\text{tot}} = 103.5\,\text{mb}$$  
     - Ensemble T, $$m_{\text{log}}(q^2)$$: $$\sigma_{\text{tot}} = 111.3\,\text{mb}$$  
     - Ensemble T, $$m_{\text{pl}}(q^2)$$: $$\sigma_{\text{tot}} = 110.9\,\text{mb}$$

2. **Dynamical Gluon Mass**:
   - Two types of dynamical gluon masses are considered: logarithmic ($$m_{\text{log}}$$) and power-law ($$m_{\text{pl}}$$).
   - The gluon mass $$m_g$$ is found to be sensitive to the dataset (ATLAS or TOTEM) and the type of mass.

3. **Beta Parameter ($$\beta_0$$)**:
   - The strength of the Pomeron coupling to quarks, $$\beta_0$$, is calculated using the dynamical gluon mass.
   - For ATLAS data:  
     - $$\beta_{0,\text{ATLAS}} = 2.33^{+0.39}_{-0.30} \, \text{GeV}^{-1}$$ (logarithmic mass)  
     - $$\beta_{0,\text{ATLAS}} = 2.13^{+0.33}_{-0.25} \, \text{GeV}^{-1}$$ (power-law mass)  
   - For TOTEM data:  
     - $$\beta_{0,\text{TOTEM}} = 2.04^{+0.28}_{-0.22} \, \text{GeV}^{-1}$$ (logarithmic mass)  
     - $$\beta_{0,\text{TOTEM}} = 1.91^{+0.22}_{-0.19} \, \text{GeV}^{-1}$$ (power-law mass)  

4. **Reggeization and Nonperturbative QCD**:
   - The scattering amplitude is Reggeized to match high-energy behavior.
   - Nonperturbative effects are incorporated through QCD effective charges and gluon propagators derived from Schwinger-Dyson equations.

## Model Parameters

### Table 1 — LN Pomeron parameters for the **logarithmic mass** $$m_{\text{log}}(q^2)$$:

| Parameter      | Ensemble A               | Ensemble T               |
|----------------|--------------------------|--------------------------|
| $$m_g$$ (GeV)  | $$0.356 \pm 0.025$$       | $$0.380 \pm 0.023$$       |
| $$\epsilon$$   | $$0.0753 \pm 0.0024$$     | $$0.0892 \pm 0.0027$$     |
| $$a_1$$ (GeV²) | $$1.373 \pm 0.017$$       | $$1.491 \pm 0.019$$       |
| $$a_2$$ (GeV²) | $$2.50 \pm 0.53$$         | $$2.77 \pm 0.60$$         |
| DoF $$\nu$$    | 88                       | 328                      |
| $$\chi^2/\nu$$ | 0.71                     | 0.67                     |

### Table 2 — LN Pomeron parameters for the **power-law mass** $$m_{\text{pl}}(q^2)$$:

| Parameter      | Ensemble A               | Ensemble T               |
|----------------|--------------------------|--------------------------|
| $$m_g$$ (GeV)  | $$0.421 \pm 0.030$$       | $$0.447 \pm 0.026$$       |
| $$\epsilon$$   | $$0.0753 \pm 0.0025$$     | $$0.0892 \pm 0.0027$$     |
| $$a_1$$ (GeV²) | $$1.517 \pm 0.019$$       | $$1.689 \pm 0.021$$       |
| $$a_2$$ (GeV²) | $$2.05 \pm 0.45$$         | $$1.70 \pm 0.51$$         |
| DoF $$\nu$$    | 88                       | 328                      |
| $$\chi^2/\nu$$ | 0.64                     | 0.90                     |

## References

The study is based on the original paper:  
G. B. Bopsin et al., *Nonperturbative gluon exchange in $$pp$$ elastic scattering at TeV energies*.  

---

**Note**: This repository is a work in progress. Contributions and collaborations are welcome!
