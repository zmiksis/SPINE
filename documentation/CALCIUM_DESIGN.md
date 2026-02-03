# Calcium Exchange Mechanism Architecture

## Design Philosophy

SPINE's calcium dynamics are governed by plasma membrane (PM) and endoplasmic reticulum (ER) exchange mechanisms. The current architecture uses a **functional programming approach** where each mechanism is implemented as a standalone function. This document describes the existing system and provides a roadmap for potential extensibility improvements.

## Key Concepts

1. **Cytosolic Calcium (C)** - Free Ca²⁺ in cytoplasm (primary variable)
2. **ER Calcium (CE)** - Free Ca²⁺ in ER lumen
3. **Buffers** - Calbindin (cytosol), calreticulin (ER)
4. **Fluxes** - Exchange rates across PM and ER membrane (umol/(um²·s))
5. **Leak Balancing** - Self-adjusting leak maintains equilibrium at rest

## 1. Plasma Membrane Mechanisms (`cytosol.py`)

### PMCA (Plasma Membrane Ca²⁺ ATPase)

**Function**: `jP(neuron, c)`

**Biophysics**: High-affinity, low-capacity pump. Hill coefficient = 2 (cooperative binding).

**Equation**:
```
J_PMCA = ρ_PMCA * I_PMCA * [Ca²⁺]² / (K_PMCA² + [Ca²⁺]²)
```

**Parameters**:
```python
params = {
    'Ip': 1.7e-17,      # Max current per pump (umol/s)
    'Kp': 60e-18,       # Michaelis constant (umol/um³)
    'rhop': 500.        # Pump density (um⁻²)
}
```

**Implementation**:
```python
def jP(neuron, c):
    """PMCA pumps - extrusion from cytosol."""
    v = (neuron.cytExchangeParams['rhop'] *
         neuron.cytExchangeParams['Ip'] * (c**2) /
         (neuron.cytExchangeParams['Kp']**2 + c**2))
    return v
```

**Physical Interpretation**:
- **At rest** (c ≈ 50 nM): Low activity, maintains baseline
- **During spike** (c > 1 μM): Saturates, limits effectiveness
- **Recovery**: Slow removal (minutes timescale)

---

### NCX (Na⁺/Ca²⁺ Exchanger)

**Function**: `jN(neuron, c)`

**Biophysics**: Low-affinity, high-capacity exchanger. Electrogenic (3 Na⁺ in, 1 Ca²⁺ out).

**Equation**:
```
J_NCX = ρ_NCX * I_NCX * [Ca²⁺] / (K_NCX + [Ca²⁺])
```

**Parameters**:
```python
params = {
    'In': 2.5e-15,      # Max current (umol/s)
    'Kn': 1.8e-15,      # Michaelis constant (umol/um³)
    'rhon': 15.         # Exchanger density (um⁻²)
}
```

**Implementation**:
```python
def jN(neuron, c):
    """NCX exchangers - extrusion from cytosol."""
    v = (neuron.cytExchangeParams['rhon'] *
         neuron.cytExchangeParams['In'] * c /
         (neuron.cytExchangeParams['Kn'] + c))
    return v
```

**Physical Interpretation**:
- **At rest**: Minimal activity
- **During spike**: Rapidly activates due to high K_m
- **Recovery**: Fast removal (seconds timescale)
- **Note**: Voltage-dependence not currently modeled

---

### VDCC (Voltage-Dependent Ca²⁺ Channel)

**Function**: `jVDCC(neuron, c, V, G)`

**Biophysics**: Opens during depolarization. Uses Goldman-Hodgkin-Katz (GHK) equation for Ca²⁺ current.

**Equation**:
```
J_VDCC = ρ_VDCC * I_GHK(V, [Ca²⁺]_in, [Ca²⁺]_out)

I_GHK = P_Ca * V * F² / (RT) * ([Ca²⁺]_in - [Ca²⁺]_out * exp(-VF/RT)) / (1 - exp(-VF/RT))
```

**Parameters**:
```python
params = {
    'rhovdcc': 1.       # Channel density (um⁻²)
}
```

**Implementation**:
```python
def jVDCC(neuron, c, V, G):
    """Voltage-dependent calcium channels - influx to cytosol."""
    from spine.vdcc import GHK

    co = neuron.cytConstants['co']  # External [Ca²⁺] = 2 mM

    # GHK current (negative = inward)
    I_GHK = GHK(V, c, co, G)

    v = neuron.cytExchangeParams['rhovdcc'] * (-I_GHK)
    return v
```

**GHK Implementation** (`vdcc.py`):
```python
def GHK(neuron, V, cc):
    """Goldman-Hodgkin-Katz equation for Ca²⁺.

    Args:
        V: Membrane voltage (V)
        c: Internal Ca²⁺ (umol/um³)
        co: External Ca²⁺ (umol/um³)
        G: Gating variable (dimensionless, 0-1)

    Returns:
        Current (umol/(um²·s))
    """
    if neuron.model['VDCC_type'] == 'N': pca = 3.8e1
    if neuron.model['VDCC_type'] == 'T': pca = 1.9e1
    if neuron.model['VDCC_type'] == 'L': pca = 5.7e1

    F = 96485.0   # Faraday constant (C/mol)
    R = 8.314     # Gas constant (J/(mol·K))
    T = 310.0     # Temperature (K)

    z = 2.0       # Valence of Ca²⁺

    c1 = z*Far*V/(Rgas*T)
    F = pca*c1*(cc - neuron.cytConstants['co']*np.exp(-c1))/(1. - np.exp(-c1))
    
    ind = (abs(V)<=1e-8)
    F[ind] = pca*((neuron.cytConstants['co'] - cc[ind]) - (Far/(Rgas*T))*(neuron.cytConstants['co'] + cc[ind])*V[ind])

    return F
```

**Physical Interpretation**:
- **Depolarized**: Large influx, initiates Ca²⁺ transients
- **Hyperpolarized**: Minimal flux
- **Gating**: Controlled by voltage-gated channel gating variable G

---

### Plasma Membrane Leak

**Function**: `jLP(neuron, c, V)`

**Biophysics**: Passive flux that balances other mechanisms at rest.

**Equation**:
```
J_leak = v_leak * ([Ca²⁺]_out - [Ca²⁺]_in)
```

**Implementation**:
```python
def jLP(neuron, c, V):
    """Leakage flux - maintains equilibrium."""
    v = neuron.cytConstants['vlp'] * (neuron.cytConstants['co'] - c)
    return v
```

**Leak Initialization** (`init_vel`):
```python
def init_vel(neuron, c, V, G):
    """Compute leak velocity to balance active mechanisms at rest."""

    if neuron.cytConstants['leak_const'] is None:
        # Sum all active mechanisms at equilibrium
        numerator = np.zeros_like(c)
        if neuron.model['PMCA']:
            numerator += jP(neuron, c)
        if neuron.model['NCX']:
            numerator += jN(neuron, c)
        if neuron.model['VDCC']:
            numerator -= jVDCC(neuron, c, V, G)

        # Compute leak velocity
        neuron.cytConstants['vlp'] = numerator / (neuron.cytConstants['co'] - neuron.settings.cceq)
    else:
        # Use user-specified constant
        neuron.cytConstants['vlp'] = neuron.cytConstants['leak_const']
```

**Physical Interpretation**:
- At rest: `J_PMCA + J_NCX + J_leak = J_VDCC`
- Automatically adjusts to maintain [Ca²⁺]_rest
- If mechanisms change, leak recomputes during initialization

---

## 2. ER Membrane Mechanisms (`er.py`)

### RyR (Ryanodine Receptor)

**Function**: `jRYR(neuron, c, ce, ryr_state, dt)`

**Biophysics**: Ca²⁺-induced Ca²⁺ release (CICR). 4-state Markov model (O1, O2, C1, C2).

**Equation**:
```
J_RyR = ρ_RyR * I_ref * (P_open_1 + P_open_2) * ([Ca²⁺]_ER - [Ca²⁺]_cyt)
```

**4-State Model**:
```
      k_a⁺[Ca]⁴        k_b⁺[Ca]³
  O1 ⇌ O2 ⇌ C1
  ↕               ↕
  k_c             k_c

  O1: Open state 1 (high Ca²⁺ sensitivity)
  O2: Open state 2 (low Ca²⁺ sensitivity)
  C1: Closed state 1 (inhibited by high Ca²⁺)
  C2: Closed state 2 (inactive)
```

**Rate Constants**:
```python
ka_neg = 28.8        # s⁻¹
ka_pos = 1500e60     # um¹²/(umol⁴·s)
kb_neg = 385.9       # s⁻¹
kb_pos = 1500e45     # um⁹/(umol³·s)
kc_neg = 0.1         # s⁻¹
kc_pos = 1.75        # s⁻¹
```

**Implementation**:
```python
def jRYR(neuron, c, ce, ryr_state, dt):
    """Ryanodine receptor - Ca²⁺-induced Ca²⁺ release from ER."""

    o1 = ryr_state[0,:]  # Open state 1
    o2 = ryr_state[1,:]  # Open state 2

    # Total open probability
    p_open = o1 + o2

    # Flux proportional to driving force
    v = (neuron.erExchangeParams['rhoryr'] *
         neuron.erExchangeParams['Irefryr'] *
         p_open * (ce - c))

    return v
```

**State Updates** (`update_ryr_state`):
- Uses implicit Euler for numerical stability
- Solves 3×3 linear system (C2 computed from normalization)
- Updates occur every timestep based on local [Ca²⁺]

**Physical Interpretation**:
- **Low Ca²⁺**: Mostly closed (C2 dominant)
- **Rising Ca²⁺**: Opens (O1, O2 increase) → amplifies signal
- **High Ca²⁺**: Inactivates (C1 increases) → prevents Ca²⁺ overload
- **CICR**: Positive feedback creates Ca²⁺ waves

---

### SERCA (Sarco/ER Ca²⁺ ATPase)

**Function**: `jS(neuron, c, ce)`

**Biophysics**: Pumps Ca²⁺ from cytosol into ER. Nonlinear kinetics.

**Equation**:
```
J_SERCA = ρ_SERCA * I_SERCA * [Ca²⁺]² / (K_SERCA² + [Ca²⁺]²)
```

**Parameters**:
```python
params = {
    'Is': 6.5e-30,      # Rate constant (umol²/(um³·s))
    'Ks': 180e-18,      # Michaelis constant (umol/um³)
    'rhos': 2390.       # Pump density (um⁻²)
}
```

**Implementation**:
```python
def jS(neuron, c, ce):
    """SERCA pumps - uptake into ER."""

    v = (neuron.erExchangeParams['rhos'] *
         neuron.erExchangeParams['Is'] * (c**2) /
         (neuron.erExchangeParams['Ks']**2 + c**2))

    return v
```

**Physical Interpretation**:
- **At rest**: Low activity, maintains ER Ca²⁺ store
- **After spike**: Rapidly refills ER (seconds-minutes)
- **Saturates**: At very high [Ca²⁺], provides upper bound on uptake

---

### IP3R (IP3 Receptor)

**Function**: `jIP3(neuron, p, c, ce)`

**Biophysics**: IP3 and Ca²⁺ co-activated channel. De Young-Keizer model (simplified).

**Equation**:
```
J_IP3 = ρ_IP3 * I_ref * P_open³ * ([Ca²⁺]_ER - [Ca²⁺]_cyt)

P_open = d₂ * [Ca²⁺] * [IP3] / ((d₁ + [Ca²⁺]) * ([Ca²⁺] + d₅))
```

**Parameters**:
```python
params = {
    'kp': 1e3,          # IP3 degradation rate (s⁻¹)
    'd1': 0.13e-15,     # Ca²⁺ dissociation const (umol/um³)
    'd2': 1.05e-15,     # Scaling constant
    'd3': 0.94e-15,     # Ca²⁺ inhibition const (umol/um³)
    'd5': 82.3e-18,     # IP3 dissociation const (umol/um³)
    'rhoI': 17.3,       # Receptor density (um⁻²)
    'IrefI': 1.1e-13    # Reference current (umol/s)
}
```

**Implementation**:
```python
def jIP3(neuron, p, c, ce):
    """IP3 receptors - IP3 and Ca²⁺ gated release from ER."""

    d1 = neuron.erExchangeParams['d1']
    d2 = neuron.erExchangeParams['d2']
    d5 = neuron.erExchangeParams['d5']

    # Open probability (cubic)
    numerator = d2 * c * p
    denominator = (c + d1) * (c + d5)
    pO = (numerator / denominator) ** 3

    # Flux
    v = (neuron.erExchangeParams['rhoI'] *
         neuron.erExchangeParams['IrefI'] *
         pO * (ce - c))

    return v
```

**IP3 Dynamics** (`reaction_ip3`):
```python
def reaction_ip3(neuron, p):
    """IP3 degradation kinetics.

    -k_p * (p - p_rest)
    """
    f = -neuron.erExchangeParams['kp'] * (p - neuron.settings.pr)
    return f
```

**Physical Interpretation**:
- **No IP3**: Closed, no release
- **IP3 + low Ca²⁺**: Opens, initiates release
- **IP3 + high Ca²⁺**: Bell-shaped response (inhibition at very high Ca²⁺)
- **Cooperative**: Cubic dependence → sharp threshold

---

### ER Membrane Leak

**Function**: `jLE(neuron, c, ce)`

**Biophysics**: Passive leak balances SERCA/IP3R/RyR at rest.

**Equation**:
```
J_ER_leak = v_leak_ER * ([Ca²⁺]_ER - [Ca²⁺]_cyt)
```

**Implementation**:
```python
def jLE(neuron, c, ce):
    """ER leak flux - maintains equilibrium."""
    v = neuron.erConstants['vle'] * (ce - c)
    return v
```

**Leak Initialization** (`init_vel_e`):
```python
def init_vel_e(neuron, c, ce, p, ryr_state, dt):
    """Compute ER leak velocity to balance mechanisms at rest."""

    if neuron.erConstants['leak_const'] is None:
        numerator = np.zeros_like(c)
        if neuron.model['RyR']:
            numerator -= jRYR(neuron, c, ce, ryr_state, dt)
        if neuron.model['SERCA']:
            numerator += jS(neuron, c, ce)
        if neuron.model['IP3']:
            numerator -= jIP3(neuron, p, c, ce)

        neuron.erConstants['vle'] = numerator / (neuron.settings.ceeq - neuron.settings.cceq)
    else:
        neuron.erConstants['vle'] = neuron.erConstants['leak_const']
```

---

## 3. Buffer Dynamics

### Cytosolic Buffer (Calbindin)

**Reaction**:
```
Ca²⁺ + Calbindin ⇌ Ca-Calbindin
         k⁺
         k⁻
```

**Equation**:
```
k⁻([B]_total - [B]) - k⁺[B][Ca²⁺]
```

**Parameters**:
```python
constants = {
    'btot': 4.0 * 40e-15,   # Total calbindin (umol/um³)
    'kbpos': 27e15,         # Forward rate (um³/(umol·s))
    'kbneg': 19.            # Reverse rate (s⁻¹)
}
```

**Implementation**:
```python
def reaction(neuron, c, b):
    """Calbindin buffering reaction."""
    f = (neuron.cytConstants['kbneg'] * (neuron.cytConstants['btot'] - b) -
         neuron.cytConstants['kbpos'] * b * c)
    return f
```

**Initialization**:
```python
def init_calb(neuron, cceq):
    """Initialize bound calbindin at equilibrium."""
    return (neuron.cytConstants['kbneg'] * neuron.cytConstants['btot'] /
            (neuron.cytConstants['kbneg'] + neuron.cytConstants['kbpos'] * cceq))
```

---

### ER Buffer (Calreticulin)

**Reaction**:
```
Ca²⁺ + Calreticulin ⇌ Ca-Calreticulin
           k⁺_ER
           k⁻_ER
```

**Parameters**:
```python
constants = {
    'betot': 4.0 * 3.6e-12,  # Total calreticulin (umol/um³)
    'kbepos': 1e14,          # Forward rate (um³/(umol·s))
    'kbeneg': 200.           # Reverse rate (s⁻¹)
}
```

**Implementation**:
```python
def reaction_er(neuron, ce, be):
    """Calreticulin buffering reaction."""
    f = (neuron.erConstants['kbeneg'] * (neuron.erConstants['betot'] - be) -
         neuron.erConstants['kbepos'] * be * ce)
    return f
```

---

## 4. Integration with Solver

### Main PDE System

The solver (`solver/helper.py`) computes:

**Cytosolic Ca²⁺**:

$$
\frac{\partial c}{\partial t} = D_c \frac{\partial^2 c}{\partial x^2} - f(c,b) + \frac{2r}{(R^2 - r^2)} J_{ER} + \frac{2R}{(R^2 - r^2)} J_{PM}
$$

**ER Ca²⁺**:

$$
\frac{\partial c_e}{\partial t} = D_{ce} \frac{\partial^2 c_e}{\partial x^2} - f(c_e,b_e) - \frac{2}{r} J_{ER} 
$$

Where:
- $D_c$, $D_{ce}$ = diffusion coefficients
- $R$ = dendrite radius
- $r$ = ER radius
- $J_{PM}$ = sum of PM fluxes
- $J_{ER}$ = sum of ER fluxes

### Flux Aggregation (`cytosol.py::JPM`, `er.py::JER`)

**Plasma Membrane**:
```python
def JPM(neuron, c, V, G, fluxes: Optional[FluxComponents] = None):
    """Aggregate all PM fluxes."""

    j = np.zeros_like(c)

    if neuron.model['PMCA']:
        jpmca = jP(neuron, c)
        j += jpmca
        if fluxes is not None and neuron.recorder.get('flux_pmca', False):
            fluxes.pmca = jpmca.copy()

    if neuron.model['NCX']:
        jncx = jN(neuron, c)
        j += jncx
        if fluxes is not None and neuron.recorder.get('flux_ncx', False):
            fluxes.ncx = jncx.copy()

    if neuron.model['VDCC']:
        jvdcc = jVDCC(neuron, c, V, G)
        j -= jvdcc  # Negative sign (influx)
        if fluxes is not None and neuron.recorder.get('flux_vdcc', False):
            fluxes.vdcc = jvdcc.copy()

    if neuron.model['PM_leak']:
        jleak = jLP(neuron, c, V)
        j -= jleak
        if fluxes is not None and neuron.recorder.get('flux_pm_leak', False):
            fluxes.pm_leak = jleak.copy()

    if fluxes is not None and neuron.recorder.get('flux_total_pm', False):
        fluxes.total_pm = j.copy()

    return j
```

**ER Membrane** (similar structure):
```python
def JER(neuron, c, ce, p, ryr_state, dt, fluxes: Optional[FluxComponents] = None):
    """Aggregate all ER fluxes."""
    # Similar pattern to JPM...
    return j
```

### Flux Recording

**FluxComponents Class** (`flux_components.py`):
```python
class FluxComponents:
    """Container for transient flux data."""

    def __init__(self):
        # PM fluxes
        self.pmca = None
        self.ncx = None
        self.vdcc = None
        self.pm_leak = None
        self.total_pm = None

        # ER fluxes
        self.ryr = None
        self.serca = None
        self.ip3r = None
        self.soc = None
        self.er_leak = None
        self.total_er = None

        # IP3
        self.total_ip3 = None
```

**Usage**:
```python
# In solver loop
if neuron.recorder.get('flux_pmca', False):
    fluxes = FluxComponents()
    j_pm = JPM(neuron, c, V, G, fluxes=fluxes)

    # Record at specific nodes or all nodes
    if neuron.recorder.get('flux_nodes'):
        nodes = neuron.recorder['flux_nodes']
        record_flux_at_nodes(fluxes.pmca[nodes], t)
    else:
        record_flux_full_array(fluxes.pmca, t)
```

---

## 5. Parameter Access and Customization

### Accessing Parameters

**Cytosolic Parameters**:
```python
from spine.utils import NeuronModel

neuron = NeuronModel()

# Access/modify PM parameters
neuron.cytExchangeParams['Ip'] = 2e-17       # Increase PMCA strength
neuron.cytExchangeParams['rhop'] = 600.      # Increase PMCA density
neuron.cytExchangeParams['rhovdcc'] = 2.     # Double VDCC density

# Access constants
neuron.cytConstants['btot'] = 5e-14          # Increase calbindin
neuron.cytConstants['co'] = 2e-12            # Change external [Ca²⁺]
```

**ER Parameters**:
```python
# Access/modify ER parameters
neuron.erExchangeParams['rhoryr'] = 5.       # Increase RyR density
neuron.erExchangeParams['Irefryr'] = 5e-12   # Stronger RyR flux
neuron.erExchangeParams['rhos'] = 3000.      # Increase SERCA density

# Access ER constants
neuron.erConstants['betot'] = 2e-11          # Increase calreticulin
```

### Default Parameters

**Cytosol** (`cytosol.py::params`):
```python
params = {
    # PMCA
    'Ip': 1.7e-17,      'Kp': 60e-18,       'rhop': 500.,
    # NCX
    'In': 2.5e-15,      'Kn': 1.8e-15,      'rhon': 15.,
    # VDCC
    'rhovdcc': 1.
}
```

**ER** (`er.py::params`):
```python
params = {
    # RyR
    'rhoryr': 3.,       'Irefryr': 3.5e-12,
    # SERCA
    'Is': 6.5e-30,      'Ks': 180e-18,      'rhos': 2390.,
    # IP3R
    'kp': 1e3,          'd1': 0.13e-15,     'd2': 1.05e-15,
    'd3': 0.94e-15,     'd5': 82.3e-18,     'rhoI': 17.3,
    'IrefI': 1.1e-13
}
```
---

## 6. Usage Examples

### Example 1: Modify PMCA Strength

```python
from spine.utils import NeuronModel

neuron = NeuronModel()
neuron.model['PMCA'] = True
neuron.model['NCX'] = True

# Increase PMCA by 50%
neuron.cytExchangeParams['Ip'] *= 1.5
neuron.cytExchangeParams['rhop'] *= 1.5

# Decrease NCX by 30%
neuron.cytExchangeParams['In'] *= 0.7
```

---

### Example 2: Disable Mechanism

```python
# Turn off VDCC
neuron.model['VDCC'] = False

# Or set density to zero
neuron.cytExchangeParams['rhovdcc'] = 0.0
```

---

### Example 3: Record Flux

```python
# Record PMCA flux at specific nodes
neuron.recorder['flux_pmca'] = True
neuron.recorder['flux_nodes'] = [0, 100, 500]

# Record all ER fluxes at all nodes (memory intensive!)
neuron.recorder['flux_ryr'] = True
neuron.recorder['flux_serca'] = True
neuron.recorder['flux_ip3r'] = True
# flux_nodes not set → records at all nodes
```

---

### Example 4: Custom Leak Constant

```python
# Override automatic leak balancing
neuron.cytConstants['leak_const'] = 1e-6  # Set manually
neuron.erConstants['leak_const'] = 5e-7

# Initialize as usual - leak will use these values
```