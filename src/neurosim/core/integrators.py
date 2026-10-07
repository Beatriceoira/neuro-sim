"""
Numerical integration methods for the biological neuron simulator.

This module implements various numerical integration schemes for solving
the differential equations that govern neuronal dynamics.
"""

from typing import Callable, Tuple, Any
import numpy as np


def euler_step(
    y: Any,
    t: float,
    dt: float,
    deriv_func: Callable[[float, Any], Any]
) -> Any:
    """
    Perform a single Euler integration step.

    y_{n+1} = y_n + dt * f(t_n, y_n)

    Args:
        y: Current state (can be float, numpy array, or dict of state variables)
        t: Current time
        dt: Time step
        deriv_func: Function that computes dy/dt = f(t, y)

    Returns:
        Updated state after Euler step
    """
    dy_dt = deriv_func(t, y)

    if isinstance(y, dict):
        new_y = {}
        for key, val in y.items():
            if key in dy_dt:
                new_y[key] = val + dt * dy_dt[key]
            else:
                new_y[key] = val
        return new_y
    else:
        return y + dt * dy_dt


def rk4_step(
    y: Any,
    t: float,
    dt: float,
    deriv_func: Callable[[float, Any], Any]
) -> Any:
    """
    Perform a single RK4 integration step.

    Args:
        y: Current state
        t: Current time
        dt: Time step
        deriv_func: Function that computes dy/dt = f(t, y)

    Returns:
        Updated state after RK4 step
    """
    k1 = deriv_func(t, y)

    if isinstance(y, dict):
        k2 = deriv_func(t + dt/2, {k: v + (dt/2) * k1[k] for k, v in y.items() if k in k1})
        k3 = deriv_func(t + dt/2, {k: v + (dt/2) * k2[k] for k, v in y.items() if k in k2})
        k4 = deriv_func(t + dt, {k: v + dt * k3[k] for k, v in y.items() if k in k3})

        new_y = {}
        for key, val in y.items():
            if key in k1:
                new_y[key] = val + (dt/6) * (k1[key] + 2*k2[key] + 2*k3[key] + k4[key])
            else:
                new_y[key] = val
        return new_y
    else:
        k2 = deriv_func(t + dt/2, y + (dt/2) * k1)
        k3 = deriv_func(t + dt/2, y + (dt/2) * k2)
        k4 = deriv_func(t + dt, y + dt * k3)
        return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


def exponential_euler_step(
    y: float,
    t: float,
    dt: float,
    target_y: float,
    tau: float
) -> float:
    """
    Perform an exponential Euler step for linear ODEs of form:
    dy/dt = (target_y - y) / tau

    Solution: y(t+dt) = target_y - (target_y - y(t)) * exp(-dt/tau)

    Useful for gating variables.
    """
    if tau == 0:
        return target_y
    return target_y - (target_y - y) * np.exp(-dt / tau)
