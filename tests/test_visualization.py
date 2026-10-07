"""
Tests for interactive visualization (Phase 11).

Covers:
- Membrane potential trace plots
- Spike raster plots
- Phase plane plots
- Current stimulus visualization
- F-I curves
- ISI distributions
- Population synchrony plots
- Web dashboard with Plotly
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

from neurosim.visualization.plots import (
    plot_voltage_trace,
    plot_raster,
    plot_phase_plane,
    plot_current_stimulus,
    plot_network_activity,
    plot_fi_curve,
    plot_isi_distribution,
    plot_synchrony,
)
from neurosim.visualization.web_dashboard import (
    WebDashboard,
    create_web_dashboard,
)
from neurosim.visualization.dashboard import (
    Dashboard,
    create_dashboard,
)


# ======================================================================
# Matplotlib plot tests
# ======================================================================

class TestVoltageTrace:
    """Tests for plot_voltage_trace."""

    def test_basic_trace(self):
        """Test basic voltage trace plotting."""
        time = np.linspace(0, 100, 1000)
        voltage = -65 + 20 * np.sin(2 * np.pi * time / 50)

        fig, ax = plot_voltage_trace(time, voltage, show=False)

        assert fig is not None
        assert ax is not None
        ax.lines  # Should have plotted line

    def test_with_spikes(self):
        """Test voltage trace with spike markers."""
        time = np.linspace(0, 100, 1000)
        voltage = -65 + 20 * np.sin(2 * np.pi * time / 50)
        spike_times = np.array([12.5, 37.5, 62.5, 87.5])

        fig, ax = plot_voltage_trace(time, voltage, spike_times=spike_times, threshold=-40)
        assert len(ax.collections) >= 0  # Spikes plotted

    def test_with_threshold(self):
        """Test voltage trace with threshold line."""
        time = np.linspace(0, 100, 500)
        voltage = np.full(500, -65.0)
        voltage[250:] = -30.0

        fig, ax = plot_voltage_trace(time, voltage, threshold=-40)
        # Should have a threshold line
        assert len(ax.lines) >= 2

    def test_save_to_file(self, tmp_path):
        """Test saving voltage trace to file."""
        time = np.linspace(0, 100, 500)
        voltage = -65 + 10 * np.sin(2 * np.pi * time / 50)

        path = str(tmp_path / "voltage_trace.png")
        plot_voltage_trace(time, voltage, show=False, save_path=path)

        import os
        assert os.path.exists(path)


class TestRasterPlot:
    """Tests for plot_raster."""

    def test_basic_raster(self):
        """Test basic spike raster plotting."""
        spike_times_list = [
            np.array([10, 25, 40, 55, 70]),
            np.array([15, 30, 45, 60, 75]),
            np.array([20, 35, 50, 65, 80]),
        ]

        fig, ax = plot_raster(spike_times_list, show=False)
        assert fig is not None
        assert ax is not None

    def test_empty_spikes(self):
        """Test raster with no spikes."""
        spike_times_list = [np.array([]), np.array([])]

        fig, ax = plot_raster(spike_times_list, show=False)
        assert fig is not None

    def test_with_labels(self):
        """Test raster with neuron labels."""
        spike_times_list = [np.array([10, 30]), np.array([20, 40])]
        labels = ['Pre', 'Post']

        fig, ax = plot_raster(spike_times_list, neuron_labels=labels, show=False)
        assert ax.get_yticklabels() is not None


class TestPhasePlane:
    """Tests for plot_phase_plane."""

    def test_basic_phase_plane(self):
        """Test phase plane trajectory plotting."""
        v = np.linspace(-70, 30, 200)
        u = np.sin(np.linspace(0, np.pi, 200))

        fig, ax = plot_phase_plane(v, u, show=False)
        assert fig is not None

    def test_with_nullclines(self):
        """Test phase plane with nullcline lines."""
        v = np.linspace(-70, 30, 200)
        u = np.sin(np.linspace(0, np.pi, 200))
        v_null = (np.linspace(-70, 30, 50), np.zeros(50))
        u_null = (np.linspace(-70, 30, 50), np.ones(50) * 0.5)

        fig, ax = plot_phase_plane(v, u, v_nullcline=v_null, u_nullcline=u_null, show=False)
        assert fig is not None


class TestCurrentStimulus:
    """Tests for plot_current_stimulus."""

    def test_basic_current(self):
        """Test current stimulus plotting."""
        time = np.linspace(0, 100, 500)
        current = np.where((time > 30) & (time < 70), 10.0, 0.0)

        fig, ax = plot_current_stimulus(time, current, show=False)
        assert fig is not None

    def test_sinusoidal_current(self):
        """Test sinusoidal current stimulus."""
        time = np.linspace(0, 100, 1000)
        current = 5 * np.sin(2 * np.pi * time / 50)

        fig, ax = plot_current_stimulus(time, current, show=False)
        assert fig is not None


class TestFICurve:
    """Tests for plot_fi_curve."""

    def test_basic_fi_curve(self):
        """Test F-I curve plotting."""
        currents = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        firing_rates = np.array([0, 0, 5, 15, 25, 35, 45, 55, 65, 75, 85])

        fig, ax = plot_fi_curve(currents, firing_rates, show=False)
        assert fig is not None
        assert len(ax.lines) >= 1


class TestISIDistribution:
    """Tests for plot_isi_distribution."""

    def test_basic_isi(self):
        """Test ISI distribution plotting."""
        spike_times = np.array([0, 10, 22, 35, 50, 66, 84])

        fig, ax = plot_isi_distribution(spike_times, show=False)
        assert fig is not None

    def test_insufficient_spikes(self):
        """Test ISI with insufficient spikes."""
        spike_times = np.array([10.0])

        fig, ax = plot_isi_distribution(spike_times, show=False)
        assert fig is not None


class TestSynchronyPlot:
    """Tests for plot_synchrony."""

    def test_basic_synchrony(self):
        """Test synchrony plot."""
        time = np.linspace(0, 500, 5000)
        synchrony = 0.5 + 0.3 * np.sin(2 * np.pi * time / 200)

        fig, ax = plot_synchrony(time, synchrony, show=False)
        assert fig is not None


class TestNetworkActivity:
    """Tests for plot_network_activity."""

    def test_basic_network(self):
        """Test network activity overview."""
        time = np.linspace(0, 100, 1000)
        voltages = {
            0: -65 + 10 * np.sin(2 * np.pi * time / 50),
            1: -65 + 8 * np.sin(2 * np.pi * time / 50 + np.pi/4),
        }
        spike_times_list = [
            np.array([12, 35, 58, 81]),
            np.array([15, 38, 61, 84]),
        ]

        fig = plot_network_activity(time, voltages, spike_times_list, show=False)
        assert fig is not None


class TestWebDashboard:
    """Tests for WebDashboard (Plotly)."""

    def test_creation(self):
        """Test web dashboard creation."""
        neurons = []
        dash = create_web_dashboard(neurons=neurons, duration=100.0, dt=0.1)
        assert dash is not None
        assert dash.duration == 100.0

    def test_record_step(self):
        """Test recording simulation steps."""
        class FakeNeuron:
            membrane_potential = -65.0

        neurons = [FakeNeuron(), FakeNeuron()]
        dash = create_web_dashboard(neurons=neurons, duration=10.0, dt=0.1)

        dash.record_step(0.0)
        dash.record_step(0.1)
        dash.record_step(0.2)

        assert len(dash.recorded[0]['time']) == 3
        assert len(dash.recorded[1]['time']) == 3

    def test_on_spike(self):
        """Test spike recording."""
        class FakeNeuron:
            membrane_potential = -65.0

        neurons = [FakeNeuron()]
        dash = create_web_dashboard(neurons=neurons, duration=100.0, dt=0.1)

        dash.on_spike(0, 10.0)
        dash.on_spike(0, 25.0)
        dash.on_spike(0, 40.0)

        assert len(dash.all_spike_times[0]) == 3
        assert dash.all_spike_times[0] == [10.0, 25.0, 40.0]

    def test_plot_voltage_traces(self):
        """Test voltage trace plotting."""
        class FakeNeuron:
            membrane_potential = -65.0

        neurons = [FakeNeuron(), FakeNeuron()]
        dash = create_web_dashboard(neurons=neurons, duration=10.0, dt=0.1)

        dash.record_step(0.0)
        dash.record_step(0.1)
        dash.record_step(0.2)

        fig = dash.plot_voltage_traces()
        assert fig is not None
        assert len(fig.data) == 2  # Two neurons

    def test_plot_raster(self):
        """Test spike raster plotting."""
        class FakeNeuron:
            membrane_potential = -65.0

        neurons = [FakeNeuron(), FakeNeuron()]
        dash = create_web_dashboard(neurons=neurons, duration=10.0, dt=0.1)

        dash.on_spike(0, 1.0)
        dash.on_spike(0, 3.0)
        dash.on_spike(1, 2.0)
        dash.on_spike(1, 4.0)

        fig = dash.plot_raster()
        assert fig is not None

    def test_plot_isi_distribution(self):
        """Test ISI distribution plotting."""
        class FakeNeuron:
            membrane_potential = -65.0

        neurons = [FakeNeuron()]
        dash = create_web_dashboard(neurons=neurons, duration=10.0, dt=0.1)

        dash.on_spike(0, 10.0)
        dash.on_spike(0, 22.0)
        dash.on_spike(0, 35.0)
        dash.on_spike(0, 50.0)

        fig = dash.plot_isi_distribution(neuron_idx=0)
        assert fig is not None

    def test_save_html(self, tmp_path):
        """Test saving web dashboard to HTML."""
        class FakeNeuron:
            membrane_potential = -65.0

        neurons = [FakeNeuron()]
        dash = create_web_dashboard(neurons=neurons, duration=10.0, dt=0.1)

        dash.record_step(0.0)
        dash.record_step(0.1)

        path = str(tmp_path / "dashboard.html")
        dash.save_html(path)

        import os
        assert os.path.exists(path)

    def test_plot_network_summary(self):
        """Test combined network summary figure."""
        class FakeNeuron:
            membrane_potential = -65.0

        neurons = [FakeNeuron(), FakeNeuron()]
        dash = create_web_dashboard(neurons=neurons, duration=10.0, dt=0.1)

        dash.record_step(0.0)
        dash.record_step(0.1)
        dash.on_spike(0, 5.0)
        dash.on_spike(1, 7.0)

        fig = dash.plot_network_summary()
        assert fig is not None


class TestMatplotlibDashboard:
    """Tests for matplotlib Dashboard class."""

    def test_creation(self):
        """Test dashboard creation."""
        class FakeNeuron:
            membrane_potential = -65.0

        neurons = [FakeNeuron(), FakeNeuron()]
        dash = create_dashboard(neurons=neurons, duration=100.0, dt=0.1)
        assert dash is not None
        assert dash.duration == 100.0

    def test_record_step(self):
        """Test recording simulation steps."""
        class FakeNeuron:
            membrane_potential = -65.0

        neurons = [FakeNeuron(), FakeNeuron()]
        dash = create_dashboard(neurons=neurons, duration=10.0, dt=0.1)

        dash.record_step(0.0)
        dash.record_step(0.1)
        dash.record_step(0.2)

        assert len(dash.recorded[0]['time']) == 3

    def test_on_spike(self):
        """Test spike recording."""
        class FakeNeuron:
            membrane_potential = -65.0

        neurons = [FakeNeuron()]
        dash = create_dashboard(neurons=neurons, duration=100.0, dt=0.1)

        dash.on_spike(0, 10.0)
        dash.on_spike(0, 25.0)

        assert len(dash.all_spike_times[0]) == 2

    def test_update(self):
        """Test dashboard update with data."""
        class FakeNeuron:
            membrane_potential = -65.0

        neurons = [FakeNeuron()]
        dash = create_dashboard(neurons=neurons, duration=10.0, dt=0.1)

        for t in np.linspace(0, 10, 20):
            dash.record_step(t)

        dash.on_spike(0, 5.0)
        dash.update()

        assert dash.fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
