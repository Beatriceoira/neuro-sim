# Neuro-Sim

A modular computational neuroscience simulator for biological neuron dynamics, synaptic interactions, and neural network activity. Supports models from leaky integrate-and-fire to full Hodgkin–Huxley conductance-based neurons with multi-compartment morphology, plasticity, and scientific visualization.

---

## Overview

Neuro-Sim provides a layered simulation pipeline:

```
stimulus ──► membrane dynamics ──► ion channels ──► action potentials
      │                                        │
      ▼                                        ▼
synaptic currents ◄── spike transmission ───► plasticity
                        │
                        ▼
                   neural populations
                        │
                        ▼
                   analysis & visualization
```

The simulator covers four levels of abstraction:

| Level | Model | Use Case |
|---|---|---|
| Simplified | LIF, Adaptive LIF | Large-scale network studies, learning rules |
| Phenomenological | Izhikevich | Fast multi-pattern spiking simulations |
| Biophysical | Hodgkin–Huxley | Single-neuron electrophysiology, ion channel dynamics |
| Spatial | Multi-compartment | Dendritic integration, morphological effects |

---

## Features

| Component | Status | Description |
|---|---|---|
| LIF Neuron | Implemented | Leaky integrate-and-fire with exponential dynamics |
| Adaptive LIF | Implemented | Spike-frequency adaptation via recovery current |
| Izhikevich | Implemented | Two-variable model with programmable firing patterns |
| Hodgkin–Huxley | Implemented | Conductance-based model with Na⁺, K⁺, and leak currents |
| Ion Channels | Implemented | Voltage-gated sodium, potassium, and leak channels |
| Excitatory Synapse | Implemented | AMPA-like conductance-based synapses |
| Inhibitory Synapse | Implemented | GABA-like conductance-based synapses |
| Short-Term Plasticity | Implemented | Tsodyks–Markram facilitation/depression model |
| STDP | Implemented | Spike-timing-dependent plasticity with exponential learning windows |
| Network Simulation | Implemented | Population-based architecture with configurable connectivity |
| Multi-Compartment | Implemented | Cable equation across morphology trees (soma, dendrites) |
| SWC Loading | Implemented | Read neuronal morphology from SWC files |
| Spike Analysis | Implemented | Firing rate, ISI, CV, Fano factor, synchrony |
| Visualization | Implemented | Matplotlib traces, rasters, phase planes; Plotly web dashboards |
| CLI | Implemented | `neurosim` command-line interface |
| JIT Acceleration | Experimental | Numba-accelerated spike detection and synaptic accumulation (falls back to NumPy) |
| Interactive Dashboard | Implemented | Matplotlib-based real-time plotting |
| Calcium Dynamics | Not implemented | — |
| NMDA Synapses | Not implemented | — |
| GPU Acceleration | Not implemented | — |

---

## Scientific Models

### Leaky Integrate-and-Fire (LIF)

The LIF model describes subthreshold membrane potential integration with exponential decay toward a resting potential:

$$\tau_m \frac{dV}{dt} = -(V - E_{leak}) + R_m I(t)$$

Spike threshold crossing triggers an instantaneous reset:

$$V \rightarrow V_{reset}$$

**State variables:** `V` (membrane potential)

**Parameters:** `Rm`, `τm`, `V_rest`, `V_th`, `V_reset`, refractory period

**Location:** `src/neurosim/neurons/lif.py`

---

### Adaptive LIF

Extends the LIF model with a spike-triggered adaptation current `w` that produces spike-frequency adaptation:

$$\tau_m \frac{dV}{dt} = -(V - E_{leak}) + R_m I(t) - w$$

$$\tau_w \frac{dw}{dt} = a(V - E_{leak}) - w$$

On each spike, `w` receives an additive increment.

**State variables:** `V`, `w` (adaptation current)

**Parameters:** `a` (subthreshold adaptation), `τ_w`, `w_increment` (spike-triggered)

**Location:** `src/neurosim/neurons/adaptive_lif.py`

---

### Izhikevich

The Izhikevich model captures diverse firing patterns with two variables:

$$\frac{dV}{dt} = 0.04V^2 + 5V + 140 - u + I$$

$$\frac{du}{dt} = a(bV - u)$$

When `V ≥ 30 mV`: `V ← c`, `u ← u + d`

**Firing pattern presets:** regular spiking, intrinsically bursting, chattering, fast spiking, low-threshold spiking

**State variables:** `V` (membrane potential), `u` (recovery variable)

**Parameters:** `a`, `b`, `c`, `d`

**Location:** `src/neurosim/neurons/izhikevich.py`

---

### Hodgkin–Huxley

The canonical conductance-based model describes action potential generation through voltage-gated ion channels:

$$C_m \frac{dV}{dt} = I_{ext} - I_{Na} - I_K - I_L$$

$$I_{Na} = g_{Na} \, m^3 \, h \, (V - E_{Na})$$

$$I_K = g_{K} \, n^4 \, (V - E_{K})$$

$$I_L = g_{L} \, (V - E_{L})$$

Gating variables follow first-order kinetics:

$$\frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)x \qquad x \in \{m, h, n\}$$

where the rate constants are:

| Gate | α(V) | β(V) |
|---|---|---|
| m | 0.1(V+40)/(1−exp(−(V+40)/10)) | 4·exp(−(V+65)/18) |
| h | 0.07·exp(−(V+65)/20) | 1/(1+exp(−(V+35)/10)) |
| n | 0.01(V+55)/(1−exp(−(V+55)/10)) | 0.125·exp(−(V+65)/80) |

**Default parameters:** C_m = 1 µF/cm², g_Na = 120 mS/cm², E_Na = +50 mV, g_K = 36 mS/cm², E_K = −77 mV, g_L = 0.3 mS/cm², E_L = −54.387 mV

**Initial conditions:** V = −65 mV, m = 0.05, h = 0.6, n = 0.32

**Location:** `src/neurosim/neurons/hodgkin_huxley.py`

---

### Ion Channels

Ion channels are implemented as pluggable components with a shared interface:

```python
class IonChannel(ABC):
    def current(self, voltage: float, state: Dict[str, float]) -> float:
        ...
    def derivatives(self, voltage: float, state: Dict[str, float]) -> Dict[str, float]:
        ...
```

Implemented channels:

| Channel | ID | Description |
|---|---|---|
| SodiumChannel | `na` | Voltage-gated Na⁺ with m³h conductance |
| PotassiumChannel | `k` | Voltage-gated K⁺ with n⁴ conductance |
| LeakChannel | `leak` | Constant leak conductance |

**Location:** `src/neurosim/channels/`

---

### Synapses

Conductance-based synapses use a dual-exponential model for synaptic conductance:

$$g_{syn}(t) = g_{peak} \left( e^{-(t-t_{spike})/\tau_{decay}} - e^{-(t-t_{spike})/\tau_{rise}} \right)$$

$$I_{syn} = g_{syn} \cdot w \cdot (V - E_{syn})$$

| Synapse | E_syn (mV) | Description |
|---|---|---|
| ExcitatorySynapse | 0 | AMPA-like glutamatergic |
| InhibitorySynapse | −70 | GABAergic |
| ShortTermPlasticitySynapse | configurable | Tsodyks–Markram facilitation/depression |
| STDSynapse | configurable | Spike-timing-dependent plasticity |

**STDP learning rule:**

$$\Delta w = \begin{cases}
+A_+ \, e^{-\Delta t / \tau_+} & \Delta t > 0 \text{ (pre before post)} \\
-A_- \, e^{\Delta t / \tau_-} & \Delta t < 0 \text{ (post before pre)}
\end{cases}$$

Weight bounds `[weight_min, weight_max]` clamp updates.

**Location:** `src/neurosim/synapses/`

---

### Network Simulation

Populations of neurons are connected with configurable rules:

```python
net = Network(dt=0.1)
exc = net.add_population("exc", LIFNeuron, size=50)
net.random_connect("exc", "exc", connection_probability=0.1, ...)
net.run(duration=500.0)
```

Connectivity options: `all_to_all_connect`, `random_connect` with probability-based rules. Synaptic delays are handled via a spike queue per connection.

**Location:** `src/neurosim/networks/network.py`

---

### Multi-Compartment Models

Neurons with dendritic trees are modeled using the cable equation. Each compartment obeys:

$$C_m \frac{dV_i}{dt} = -I_{membrane}^{(i)} + \sum_j \frac{V_j - V_i}{R_{ij}} + I_{ext}^{(i)}$$

Axial resistance between adjacent compartments:

$$R_{ij} = \frac{R_a}{2} \left( \frac{L_i}{\pi r_i^2} + \frac{L_j}{\pi r_j^2} \right)$$

Morphology can be built programmatically or loaded from SWC files.

**Location:** `src/neurosim/neurons/multi_compartment.py`, `compartment.py`, `morphology.py`

---

## Architecture

```
src/neurosim/
├── core/            Simulation engine, state management, numerical integrators
│   ├── simulation.py     Main Simulation class
│   ├── state.py          NeuronState, NetworkState, SimulationState dataclasses
│   ├── integrators.py    Euler, RK4, exponential-Euler integration
│   └── events.py         SpikeEvent, SynapticEvent, EventQueue
├── neurons/         Neuron models
│   ├── lif.py              Leaky integrate-and-fire
│   ├── adaptive_lif.py     Adaptation current model
│   ├── izhikevich.py       Two-variable spiking model
│   ├── hodgkin_huxley.py   Full conductance-based model
│   ├── multi_compartment.py Cable equation across morphology trees
│   ├── compartment.py      Individual electrical compartment
│   └── morphology.py       Tree structure (soma → dendrites), SWC loader
├── channels/        Ion channel models
│   ├── base.py               Abstract IonChannel interface
│   ├── sodium.py             g_Na m³h(V−E_Na)
│   ├── potassium.py          g_K n⁴(V−E_K)
│   ├── leak.py               g_L(V−E_L)
│   └── channel_models.py     ChannelManager, factory functions
├── synapses/        Synaptic models
│   ├── base.py                  Abstract BaseSynapse
│   ├── excitatory.py            AMPA-like
│   ├── inhibitory.py            GABA-like
│   ├── stp.py                   Tsodyks–Markram short-term plasticity
│   └── stdp.py                  Spike-timing-dependent plasticity
├── stimuli/         Input current generators
│   ├── base.py                Abstract BaseStimulus
│   ├── step.py                Rectangular current pulse
│   └── sinusoidal.py          Sinusoidal current
├── networks/        Population and connectivity
│   └── network.py              Network, Connection classes
├── analysis/        Spike train analysis
│   └── spikes.py              Firing rate, ISI, CV, Fano factor, synchrony
├── visualization/   Plotting utilities
│   ├── plots.py               Matplotlib: traces, rasters, phase planes
│   ├── dashboard.py           Real-time matplotlib dashboard
│   └── web_dashboard.py       Plotly-based interactive web dashboard
├── optimization/    JIT-accelerated hot loops
│   └── jit.py                 Numba-accelerated spike detection and synaptic accumulation
└── cli.py           Command-line interface
```

---

## Installation

```bash
git clone <repository-url>
cd neuro-sim
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### Core dependencies

| Package | Purpose |
|---|---|
| NumPy ≥ 1.24 | Array computation |
| SciPy ≥ 1.10 | Scientific utilities |
| Matplotlib ≥ 3.7 | Static plotting |
| Pandas ≥ 2.0 | Data analysis |
| Plotly ≥ 5.15 | Interactive web visualizations |
| Dash ≥ 2.15 | Web dashboard framework |
| PyYAML ≥ 6.0 | Experiment configuration |
| pytest ≥ 7.0 | Test runner |

### Optional development dependencies

```bash
pip install -e ".[dev]"
```

Includes `black`, `flake8`, `mypy`, and `pre-commit`.

### Package discovery

The project uses a `src` layout. `pyproject.toml` configures setuptools to discover packages under `src/`:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

### JIT acceleration (optional)

Numba is auto-detected at import time. If available, the optimization module uses `@njit` compilation; otherwise it falls back to NumPy.

```bash
pip install numba
```

---

## Quick Start

```bash
PYTHONPATH=src python examples/run_lif_simple.py
```

Or programmatically:

```python
from neurosim.neurons import LIFNeuron
from neurosim.stimuli import StepCurrent
from neurosim.analysis.spikes import analyze_spike_train
import numpy as np

neuron = LIFNeuron(
    membrane_resistance=10.0,    # MΩ
    membrane_time_constant=20.0, # ms
    resting_potential=-65.0,     # mV
    threshold_potential=-50.0,   # mV
    reset_potential=-65.0,       # mV
    refractory_period=2.0        # ms
)

stimulus = StepCurrent(amplitude=10.0, start_time=50.0, end_time=150.0)

for i in range(3000):
    t = float(i) * 0.1
    neuron.update(t, dt=0.1, external_current=stimulus.get_current(t))

spikes = np.array(neuron.state.spike_times)
print(f"Spikes: {len(spikes)}, firing rate: {len(spikes)/2.0:.1f} Hz")
```

---

## Running Experiments

### Via CLI

```bash
# List available models
neurosim list-models

# Run tests
neurosim test

# Help
neurosim --help
```

### Experiment configuration

YAML experiment files define neuron models, stimuli, and simulation parameters:

```yaml
# experiments/basic_lif.yaml
neuron:
  model: lif
  parameters:
    membrane_resistance: 10.0
    membrane_time_constant: 20.0
    resting_potential: -65.0
    threshold_potential: -50.0
    reset_potential: -65.0
    refractory_period: 2.0

simulation:
  duration: 200.0
  dt: 0.01
  spike_threshold: -50.0
  spike_reset: -65.0
  refractory_period: 2.0

stimulus:
  type: step
  amplitude: 10.0
  start_time: 50.0
  end_time: 150.0

seed: 42
```

---

## Visualization

### Matplotlib plots

```python
from neurosim.visualization.plots import (
    plot_voltage_trace,
    plot_raster,
    plot_phase_plane,
    plot_fi_curve,
    plot_isi_distribution,
)
```

| Function | Description |
|---|---|
| `plot_voltage_trace` | Membrane potential over time with spike markers and threshold |
| `plot_raster` | Spike raster for population of neurons |
| `plot_phase_plane` | V vs. recovery variable trajectory with nullclines |
| `plot_current_stimulus` | Input current waveform |
| `plot_network_activity` | Combined voltage traces + raster overview |
| `plot_fi_curve` | Firing rate vs. injected current |
| `plot_isi_distribution` | Inter-spike interval histogram |
| `plot_synchrony` | Population synchrony index over time |

### Interactive dashboards

```python
from neurosim.visualization import Dashboard, WebDashboard

# Matplotlib-based real-time dashboard
dash = Dashboard(neurons=[neuron], duration=200.0, dt=0.1)

# Plotly web dashboard
web_dash = WebDashboard(neurons=[neuron], duration=200.0, dt=0.1)
web_dash.save_html("dashboard.html")
```

---

## Analysis

Spike train analysis tools are in `src/neurosim/analysis/spikes.py`:

| Function | Description |
|---|---|
| `analyze_spike_train` | Firing rate, mean/std ISI, CV, Fano factor |
| `spike_times_to_intervals` | Convert spike times to inter-spike intervals |
| `mean_firing_rate` | Average firing rate in Hz |
| `coefficient_of_variation` | ISI variability (CV = σ/μ) |
| `detect_spikes_from_voltage` | Threshold-crossing detection with refractory filter |
| `population_synchrony` | Variance/mean of binned population spike counts |
| `cross_correlation` | Cross-correlogram between two spike trains |

---

## Reproducibility

- The project uses fixed timesteps for deterministic integration.
- Stochastic components (e.g., random connectivity) accept a `seed` parameter in configuration files.
- Experiment YAML files pin simulation duration, timestep, and seed for reproducible runs.
- The `optimization/jit.py` module auto-detects Numba availability and provides a pure-NumPy fallback.

---

## Examples

| Example | Description |
|---|---|
| `examples/run_lif_simple.py` | LIF neuron with step current |
| `examples/run_basic_lif.py` | LIF with visualization and spike analysis |
| `examples/run_multi_compartment.py` | Multi-compartment HH neuron with SWC-style morphology |
| `examples/run_interactive_dashboard.py` | Matplotlib real-time dashboard demo |
| `examples/demo_phase_14.py` | Full feature demonstration (LIF, HH, network, analysis) |

Run any example with:

```bash
PYTHONPATH=src python examples/run_lif_simple.py
```

---

## Testing

```bash
pytest                         # Run all tests
pytest -v                      # Verbose output
pytest --cov                   # With coverage (requires pytest-cov)
```

Test suite: **94 tests** covering neuron models, multi-compartment dynamics, synapses, STDP, and visualization.

| Test file | Coverage |
|---|---|
| `tests/test_lif.py` | LIF initialization, subthreshold dynamics, spiking, refractory period |
| `tests/test_izhikevich.py` | Izhikevich parameter presets, dynamics, spiking |
| `tests/test_hodgkin_huxley.py` | HH initial conditions, ionic currents, spiking |
| `tests/test_multi_compartment.py` | Compartment geometry, morphology, cable equation, dendritic integration, SWC loading |
| `tests/test_stdp.py` | STDP LTP/LTD, weight bounds, conductance update |
| `tests/test_visualization.py` | Matplotlib and Plotly dashboard plotting |

---

## Numerical Considerations

- **Integration:** Forward Euler is the default. `integrators.py` also provides RK4 and exponential-Euler (exact solution for linear ODEs, useful for gating variables).
- **Timestep:** Recommended `dt ≤ 0.1 ms` for Hodgkin–Huxley and multi-compartment models. Larger timesteps may cause numerical instability or inaccurate spike timing.
- **Gating variable bounds:** All implemented gating variables remain within [0, 1] at tested timesteps.
- **NaN/Inf:** No uncontrolled divergence observed in unit tests. Numerical checks are performed in the benchmark suite.
- **Numba JIT:** When available, spike detection (`detect_spikes`) and synaptic accumulation (`accumulate_synaptic_currents`) are accelerated via `@njit`. Falls back to NumPy when Numba is absent.

---

## Scientific Limitations

Neuro-Sim is a **computational model**, not a complete biological reconstruction. Key limitations:

- **Morphology:** Multi-compartment models use cylindrical approximations; real dendritic trees have tapering, spines, and non-cylindrical geometry.
- **Ion channels:** Only sodium, potassium, and leak channels are implemented. No calcium, HCN, or other modulatory channels.
- **Synapses:** Simplified dual-exponential conductance model; no neurotransmitter diffusion, vesicle pool depletion (beyond STP), or NMDA voltage dependence.
- **Plasticity:** STDP uses a phenomenological exponential window; does not model NMDA-receptor dependent spike-timing mechanisms.
- **Network scale:** Python-loop integration limits simulations to hundreds–thousands of neurons on typical hardware without JIT.
- **Parameter values:** Default parameters follow canonical Hodgkin–Huxley (1952) but are not calibrated to any specific cell type.

---

## Project Structure

```
neuro-sim/
├── src/neurosim/
│   ├── __init__.py
│   ├── cli.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── events.py
│   │   ├── integrators.py
│   │   ├── simulation.py
│   │   └── state.py
│   ├── neurons/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── lif.py
│   │   ├── adaptive_lif.py
│   │   ├── izhikevich.py
│   │   ├── hodgkin_huxley.py
│   │   ├── compartment.py
│   │   ├── morphology.py
│   │   └── multi_compartment.py
│   ├── channels/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── sodium.py
│   │   ├── potassium.py
│   │   ├── leak.py
│   │   └── channel_models.py
│   ├── synapses/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── excitatory.py
│   │   ├── inhibitory.py
│   │   ├── stp.py
│   │   └── stdp.py
│   ├── stimuli/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── step.py
│   │   └── sinusoidal.py
│   ├── networks/
│   │   ├── __init__.py
│   │   └── network.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── spikes.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plots.py
│   │   ├── dashboard.py
│   │   └── web_dashboard.py
│   └── optimization/
│       ├── __init__.py
│       └── jit.py
├── tests/
│   ├── test_lif.py
│   ├── test_izhikevich.py
│   ├── test_hodgkin_huxley.py
│   ├── test_multi_compartment.py
│   ├── test_stdp.py
│   └── test_visualization.py
├── examples/
│   ├── run_lif_simple.py
│   ├── run_basic_lif.py
│   ├── run_multi_compartment.py
│   ├── run_interactive_dashboard.py
│   └── demo_phase_14.py
├── experiments/
│   └── basic_lif.yaml
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Roadmap

### Implemented
- LIF, Adaptive LIF, Izhikevich, Hodgkin–Huxley neuron models
- Sodium, potassium, and leak ion channels
- Excitatory and inhibitory conductance-based synapses
- Short-term plasticity (Tsodyks–Markram)
- Spike-timing-dependent plasticity (STDP)
- Network simulation with configurable connectivity
- Multi-compartment morphology with SWC loading
- Spike train and population analysis
- Matplotlib and Plotly visualization
- CLI entry point
- JIT-accelerated spike detection and synaptic accumulation (Numba)

### Experimental
- Numba JIT acceleration (falls back to NumPy; no GPU support)
- Real-time matplotlib dashboard

### Benchmark suite
- Numerical stability checks across parameter ranges (3 param sets tested)
- Network scaling benchmarks (sizes 10, 50, 100; durations 50/100/200 ms)
- F-I curve validation against published reference data
- Steady-state firing rate convergence checks
- Regression tests for LIF, network, HH, and Izhikevich spike counts

### Planned
- Additional ion channels (calcium, HCN, calcium-dependent potassium)
- NMDA receptor synapses with voltage-dependent Mg²⁺ block
- Larger-scale sparse network simulation
- GPU-accelerated multi-compartment integration
- Parameter fitting and model calibration tools
- Jupyter notebook tutorials

---

## Contributing

Contributions are welcome. To get started:

```bash
git clone <repository-url>
cd neuro-sim
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Write tests for new functionality under `tests/`
3. Run the test suite: `pytest`
4. Commit changes: `git commit -m "Add my feature"`
5. Push and open a pull request

---

## License

License information has not yet been specified.
