"""Tests for STDP synapse model."""
import pytest
import numpy as np
from neurosim.synapses.stdp import STDSynapse


class TestSTDSynapse:
    def test_creation(self):
        s = STDSynapse()
        assert s.weight == 1.0
        assert s.reversal_potential == 0.0

    def test_conductance_update(self):
        s = STDSynapse()
        s.update_state(pre_synaptic_spike=True, post_synaptic_spike=False, current_time=0.0, dt=0.01)
        assert s.state['g'] >= 0.0

    def test_stdp_ltp(self):
        s = STDSynapse(A_plus=0.01, A_minus=0.012)
        initial_weight = s.weight
        s.pre_synaptic_spike_times.append(10.0)
        s._apply_stdp_rule(pre_synaptic_spike=False, post_synaptic_spike=True, current_time=20.0)
        assert s.weight >= initial_weight  # LTP: pre before post

    def test_stdp_ltd(self):
        s = STDSynapse(A_plus=0.01, A_minus=0.012)
        initial_weight = s.weight
        s.post_synaptic_spike_times.append(10.0)
        s._apply_stdp_rule(pre_synaptic_spike=True, post_synaptic_spike=False, current_time=20.0)
        assert s.weight <= initial_weight  # LTD: post before pre

    def test_weight_bounds(self):
        s = STDSynapse(weight_max=2.0, weight_min=0.0)
        for _ in range(10000):
            s._apply_stdp_rule(pre_synaptic_spike=True, post_synaptic_spike=True, current_time=100.0)
        assert s.weight_min <= s.weight <= s.weight_max

    def test_no_defined_spike_name_error(self):
        s = STDSynapse()
        # This used to raise NameError: pre_synaptic_spike not defined
        s._apply_stdp_rule(pre_synaptic_spike=False, post_synaptic_spike=False, current_time=0.0)
