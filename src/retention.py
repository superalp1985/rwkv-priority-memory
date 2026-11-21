"""Capacity-triggered retention of explicit cards, not RWKV tensor slots."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Contest:
    incoming: str
    residents: tuple[str, ...]
    needed: int
    evicted: tuple[str, ...]
    tied: bool


@dataclass(frozen=True)
class Update:
    cards: tuple[str, ...]
    retired: tuple[str, ...]
    evicted: tuple[str, ...]
    deferred: dict[str, str]
    contests: tuple[Contest, ...]


def update_cards(residents, required, lengths, budget, score, *, superseded=None):
    """Apply trusted admission/version decisions; score only selectable victims.

    The caller decides what needs writing. Earlier successful writes on the same
    turn cannot be evicted by a subsequent write. A deferred write stays archived;
    an invalidated version never becomes current again just because its new card
    could not fit. Equal-score victims are evicted in original admission order.
    """
    superseded = superseded or {}
    residents = tuple(residents)
    replacements = tuple(new for new in superseded.values() if new is not None)
    required = tuple(dict.fromkeys((*replacements, *required)))
    if len(residents) != len(set(residents)) or superseded.keys() & set(required):
        raise ValueError('duplicate residency or attempt to admit a retired version')
    if type(budget) is not int or budget < 1:
        raise ValueError('budget must be a positive integer')
    if any(type(lengths[item]) is not int or lengths[item] < 1 for item in (*residents, *required)):
        raise ValueError('card lengths must be positive integer token counts')
    if sum(lengths[item] for item in residents) > budget:
        raise ValueError('resident cards exceed the configured budget')

    retired = tuple(item for item in residents if item in superseded)
    active = [item for item in residents if item not in superseded]
    used = sum(lengths[item] for item in active)
    protected = set(required) & set(active)
    evicted, deferred, contests = [], {}, []
    for item in required:
        if item in active:
            continue
        length = lengths[item]
        if length > budget:
            deferred[item] = 'oversize'
            continue
        needed = used + length - budget
        if needed > 0:
            candidates = tuple(old for old in active if old not in protected)
            victims, tied = [], False
            if sum(lengths[old] for old in candidates) < needed:
                deferred[item] = 'same_turn_capacity'
            elif len(candidates) == 1:
                victims = list(candidates)
            else:
                scores = score(candidates)
                if any(scores.get(old) is None for old in candidates):
                    deferred[item] = 'unranked_residents'
                else:
                    if not all(isfinite(scores[old]) for old in candidates):
                        raise ValueError('victim scores must be finite')
                    ranked = sorted(candidates, key=scores.__getitem__)
                    remaining = needed
                    for old in ranked:
                        victims.append(old)
                        remaining -= lengths[old]
                        if remaining <= 0:
                            break
                    if len(victims) < len(ranked):
                        tied = scores[victims[-1]] == scores[ranked[len(victims)]]
            contests.append(Contest(item, candidates, needed, tuple(victims), tied))
            if item in deferred:
                continue
            for old in victims:
                active.remove(old)
                used -= lengths[old]
            evicted.extend(victims)
        active.append(item)
        used += length
        protected.add(item)

    return Update(tuple(active), retired, tuple(evicted), deferred, tuple(contests))