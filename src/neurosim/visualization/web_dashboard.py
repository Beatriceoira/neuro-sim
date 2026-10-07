"""
Web-based interactive dashboard for biological neuron simulations.

Uses Plotly to create browser-based interactive visualizations
with real-time updates. Supports membrane potential traces,
spike rasters, phase planes, F-I curves, and network activity.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class WebDashboard:
    """
    Browser-based interactive dashboard using Plotly.

    Provides interactive plots that can be updated at each
    simulation step. Supports hover tooltips, zoom, pan,
    and export to HTML.

    Parameters
    ----------
    neurons : list
        List of neuron objects to monitor.
    duration : float
        Simulation duration (ms).
    dt : float
        Time step (ms).
    """

    def __init__(
        self,
        neurons: List[Any],
        duration: float,
        dt: float = 0.1,
    ):
        self.neurons = neurons
        self.duration = duration
        self.dt = dt
        self.n_neurons = len(neurons)

        self.recorded = {i: {'time': [], 'voltage': []} for i in range(self.n_neurons)}
        self.all_spike_times = [[] for _ in neurons]

    def record_step(self, t: float):
        """Record current state of all neurons."""
        for i, neuron in enumerate(self.neurons):
            self.recorded[i]['time'].append(t)
            self.recorded[i]['voltage'].append(neuron.membrane_potential)

    def on_spike(self, neuron_idx: int, t: float):
        """Record a spike event."""
        self.all_spike_times[neuron_idx].append(t)

    def plot_voltage_traces(self) -> go.Figure:
        """Plot membrane potential traces for all neurons."""
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        for i in range(self.n_neurons):
            rec = self.recorded[i]
            if rec['time']:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=rec['time'],
                    y=rec['voltage'],
                    mode='lines',
                    name=f'Neuron {i}',
                    line=dict(color=color, width=1.5),
                ))

        fig.update_layout(
            title='Membrane Potential Traces',
            xaxis_title='Time (ms)',
            yaxis_title='Membrane Potential (mV)',
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
            template='plotly_white',
            hovermode='x unified',
        )
        return fig

    def plot_raster(self) -> go.Figure:
        """Plot spike raster diagram."""
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        for i, spikes in enumerate(self.all_spike_times):
            if spikes:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=spikes,
                    y=[i] * len(spikes),
                    mode='markers',
                    name=f'Neuron {i}',
                    marker=dict(color=color, size=5),
                ))

        fig.update_layout(
            title='Spike Raster Plot',
            xaxis_title='Time (ms)',
            yaxis_title='Neuron Index',
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
            template='plotly_white',
        )
        return fig

    def plot_phase_plane(
        self,
        neuron_idx: int = 0,
        v_key: str = 'voltage',
        u_key: Optional[str] = None,
    ) -> go.Figure:
        """Plot phase plane trajectory for a single neuron."""
        rec = self.recorded[neuron_idx]
        v = np.array(rec[v_key])

        if u_key and u_key in rec:
            u = np.array(rec[u_key])
        else:
            # Use gating variables if available
            u = np.zeros_like(v)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=v,
            y=u,
            mode='lines',
            name='Trajectory',
            line=dict(color='blue', width=1.5),
        ))

        # Mark start and end
        fig.add_trace(go.Scatter(
            x=[v[0]], y=[u[0]],
            mode='markers',
            name='Start',
            marker=dict(color='green', size=10),
        ))
        fig.add_trace(go.Scatter(
            x=[v[-1]], y=[u[-1]],
            mode='markers',
            name='End',
            marker=dict(color='red', size=10),
        ))

        fig.update_layout(
            title=f'Phase Plane (Neuron {neuron_idx})',
            xaxis_title='Voltage (mV)',
            yaxis_title='Recovery / Gate',
            template='plotly_white',
        )
        return fig

    def plot_fi_curve(self, currents: List[float], firing_rates: List[float]) -> go.Figure:
        """Plot F-I curve (frequency vs current)."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=currents,
            y=firing_rates,
            mode='lines+markers',
            name='F-I Curve',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
        ))

        fig.update_layout(
            title='F-I Curve (Frequency vs Injected Current)',
            xaxis_title='Injected Current (pA)',
            yaxis_title='Firing Rate (Hz)',
            template='plotly_white',
        )
        return fig

    def plot_isi_distribution(self, neuron_idx: int = 0) -> go.Figure:
        """Plot inter-spike interval distribution."""
        spikes = self.all_spike_times[neuron_idx]
        if len(spikes) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient spikes (<2)", xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
            return fig

        isis = np.diff(sorted(spikes))
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=isis,
            nbinsx=20,
            name='ISI',
            marker_color='skyblue',
            opacity=0.7,
        ))

        fig.update_layout(
            title=f'ISI Distribution (Neuron {neuron_idx})',
            xaxis_title='ISI (ms)',
            yaxis_title='Count',
            template='plotly_white',
        )
        return fig

    def plot_network_summary(self) -> go.Figure:
        """Combined network overview figure."""
        n_cols = 2
        n_rows = 2
        subplot_titles = ['Voltage Traces', 'Spike Raster', 'F-I Curve', 'ISI Distribution']

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # 1. Voltage traces
        colors = ['blue', 'red', 'green', 'orange']
        for i in range(self.n_neurons):
            rec = self.recorded[i]
            if rec['time']:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=rec['time'], y=rec['voltage'],
                    mode='lines', name=f'Neuron {i}',
                    line=dict(color=color, width=1.2),
                ), row=1, col=1)

        # 2. Raster
        for i, spikes in enumerate(self.all_spike_times):
            if spikes:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=spikes, y=[i] * len(spikes),
                    mode='markers', name=f'Neuron {i}',
                    marker=dict(color=color, size=4),
                ), row=1, col=2)

        # 3. F-I curve (placeholder)
        fig.add_annotation(text="Run FI curve experiment", xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False, row=2, col=1)

        # 4. ISI distribution
        if len(self.all_spike_times[0]) >= 2:
            isis = np.diff(sorted(self.all_spike_times[0]))
            fig.add_trace(go.Histogram(
                x=isis, nbinsx=20, name='ISI',
                marker_color='skyblue', opacity=0.7,
            ), row=2, col=2)

        fig.update_layout(
            height=800,
            title_text='Network Activity Summary',
            template='plotly_white',
        )
        return fig

    def save_html(self, path: str):
        """Save dashboard as interactive HTML file."""
        fig = self.plot_network_summary()
        fig.write_html(path)

    def show(self):
        """Display the web dashboard (opens in browser)."""
        fig = self.plot_network_summary()
        fig.show()


def create_web_dashboard(
    neurons: List[Any],
    duration: float,
    dt: float = 0.1,
) -> WebDashboard:
    """Factory function to create a web dashboard."""
    return WebDashboard(neurons=neurons, duration=duration, dt=dt)
