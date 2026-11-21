"""Scalar contract for a causal post-write retention trace (no checkpoint needed)."""

import json
import math


def _unit(value):
    if not 0 <= value <= 1:
        raise ValueError('expected a finite value in [0, 1]')
    return value


def _offset(previous_trace, strength, mask):
    if not 0 <= strength <= 0.25:
        raise ValueError('strength must be in [0, 0.25] for this probe')
    return -strength * _unit(mask) * _unit(previous_trace)


def next_trace(previous, innovation, *, half_life=64):
    if not (math.isfinite(half_life) and half_life > 0):
        raise ValueError('half_life must be positive and finite')
    return max(_unit(innovation), _unit(previous) * 2**(-1.0 / half_life))


def retention(w, previous_trace, *, strength=0.1, mask=1.0):
    """Modulate effective keep factor, NOT the upstream preactivation."""
    if not 0 < w <= 1:
        raise ValueError('expected a positive keep factor at most 1')
    delta = _offset(previous_trace, strength, mask)
    return w if not delta else w**math.exp(delta)


def kernel_log_decay(z, previous_trace, *, strength=0.1, mask=1.0):
    """Algebraic argument for a kernel which computes w = exp(-exp(z))."""
    if not math.isfinite(z):
        raise ValueError('expected a finite kernel argument')
    delta = _offset(previous_trace, strength, mask)
    return z if not delta else z + delta


def main():
    w, trace, native, scheduled = 0.8, 0.0, 0.0, 0.0
    events = []
    for step in range(4):
        write = float(step == 0)
        native = native * w + write
        scheduled = scheduled * retention(w, trace) + write
        events.append({'t': step, 'write': write, 'trace_before': round(trace, 6),
                       'native_state': round(native, 6), 'scheduled_state': round(scheduled, 6)})
        trace = next_trace(trace, write)
    print(json.dumps({'scope': 'one scalar, no erase term, no model, not a recall metric',
                      'events': events}, indent=2))


if __name__ == '__main__':
    main()
