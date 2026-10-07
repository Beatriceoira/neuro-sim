"""
Interactive visualization for the biological neuron simulator (Phase 11).

Provides:
- Dashboard class for real-time simulation control
- Membrane potential trace plots
- Spike raster plots
- Phase plane plots (v vs w for Izhikevich)
- Current stimulus visualization
- Network activity views
- Plotly-based web dashboard
"""

from .dashboard import Dashboard, create_dashboard
from .plots import (
    plot_voltage_trace,
    plot_raster,
    plot_phase_plane,
    plot_current_stimulus,
    plot_network_activity,
    plot_fi_curve,
    plot_isi_distribution,
    plot_synchrony,
)
from .web_dashboard import WebDashboard, create_web_dashboard

__all__ = [
    "Dashboard",
    "create_dashboard",
    "WebDashboard",
    "create_web_dashboard",
    "plot_voltage_trace",
    "plot_raster",
    "plot_phase_plane",
    "plot_current_stimulus",
    "plot_network_activity",
    "plot_fi_curve",
    "plot_isi_distribution",
    "plot_synchrony",
]