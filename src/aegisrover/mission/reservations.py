"""Shared-resource reservations: one-at-a-time booking and batch planning.

``reserve`` keeps the original try-one/report-conflict workflow. ``plan``
accepts a whole batch of requests at once and returns a feasible schedule:
every robot gets a non-overlapping window, every contention is settled by an
explained ordering decision, and the waits are reported per robot so the
most-starved one is visible instead of anonymous.
"""
from bisect import insort
from dataclasses import dataclass
from typing import Iterable

__all__ = ('Reservation', 'conflicts', 'can_reserve', 'reserve',
           'Request', 'Slot', 'Decision', 'Plan', 'plan', 'feasible')


@dataclass(frozen=True)
class Reservation:
    robot: str
    resource: str
    start: float
    end: float

def conflicts(a: Reservation, b: Reservation):
    return a.resource == b.resource and max(a.start, b.start) < min(a.end, b.end)

def can_reserve(existing, candidate):
    return not any((conflicts(r, candidate) for r in existing if r.robot != candidate.robot))

def reserve(existing, candidate):
    if not can_reserve(existing, candidate):
        raise ValueError('conflict')
    return [*existing, candidate]


@dataclass(frozen=True)
class Request:
    """One robot's wish to hold ``resource`` for ``duration`` from ``earliest`` on.

    ``submitted`` is the queue timestamp: between equal-priority requests the
    one submitted earlier has waited longer and goes first, so a stream of
    newcomers cannot starve a request that has been queued for a while.
    """
    robot: str
    resource: str
    earliest: float
    duration: float
    priority: int = 0
    submitted: float = 0.0

    def __post_init__(self):
        if not self.robot or not self.resource:
            raise ValueError('robot and resource are required')
        if not self.duration >= 0:
            raise ValueError('duration must not be negative')

    @property
    def desired_end(self) -> float:
        return self.earliest + self.duration


@dataclass(frozen=True)
class Slot:
    """A placed window; ``wait`` is how long past the desired start it begins."""
    robot: str
    resource: str
    start: float
    end: float
    wait: float


@dataclass(frozen=True)
class Decision:
    """One settled contention: ``first`` goes ahead of ``then`` because ``reason``."""
    resource: str
    first: str
    then: str
    reason: str


@dataclass(frozen=True)
class Plan:
    """A feasible batch schedule plus its explanations and per-robot waits."""
    slots: tuple[Slot, ...]
    decisions: tuple[Decision, ...]
    waits: dict[str, float]

    @property
    def longest_waiting(self) -> tuple[str, float] | None:
        """The (robot, wait) kept waiting longest, or ``None`` for an empty plan."""
        if not self.waits:
            return None
        return min(self.waits.items(), key=lambda item: (-item[1], item[0]))

    def as_reservations(self) -> tuple[Reservation, ...]:
        """The planned slots as plain reservations, ready to commit with ``reserve``."""
        return tuple(Reservation(s.robot, s.resource, s.start, s.end) for s in self.slots)


def plan(requests: Iterable[Request], existing: Iterable[Reservation] = ()) -> Plan:
    """Schedule a batch of requests into one feasible, explained plan.

    Requests are grouped by resource and ordered by ``(-priority, submitted,
    earliest, robot)``: urgency first, then seniority so long-waiting robots
    are not starved by equal-priority newcomers. Each request gets the earliest
    free window at or after its desired start that overlaps neither ``existing``
    reservations nor slots already placed from this batch. Every interval that
    pushes a request back yields a ``Decision`` naming who goes first and why,
    and ``Plan.waits`` totals how long each robot is held past its desired
    start. Resources are planned independently: a robot's door and charger
    windows may overlap in time.
    """
    groups: dict[str, list[Request]] = {}
    for req in requests:
        groups.setdefault(req.resource, []).append(req)
    fixed: dict[str, list[Reservation]] = {}
    for res in existing:
        fixed.setdefault(res.resource, []).append(res)

    slots: list[Slot] = []
    decisions: list[Decision] = []
    waits: dict[str, float] = {}
    for resource in sorted(groups):
        placed = sorted(((r.start, r.end, r.robot, None) for r in fixed.get(resource, ())),
                        key=_window_key)
        for req in sorted(groups[resource], key=_queue_order):
            start, blockers = _place(placed, req.earliest, req.duration)
            for b_start, b_end, holder, source in blockers:
                if holder == req.robot:
                    continue
                reason = (_reason(source, req) if source is not None
                          else f'already reserved [{b_start:g}, {b_end:g})')
                decisions.append(Decision(resource, holder, req.robot, reason))
            end = start + req.duration
            insort(placed, (start, end, req.robot, req), key=_window_key)
            slots.append(Slot(req.robot, resource, start, end, start - req.earliest))
            waits[req.robot] = waits.get(req.robot, 0.0) + start - req.earliest
    ordered = tuple(sorted(slots, key=lambda s: (s.resource, s.start, s.robot)))
    return Plan(ordered, tuple(decisions), waits)


def feasible(windows, existing=()) -> bool:
    """True when no two windows on the same resource overlap.

    Windows are half-open ``[start, end)`` intervals, so back-to-back slots on
    one resource are fine. Anything with ``resource``, ``start`` and ``end``
    attributes (``Slot``, ``Reservation``) is accepted.
    """
    all_windows = [*existing, *windows]
    if any(w.end < w.start for w in all_windows):
        return False
    ordered = sorted((w.resource, w.start, w.end) for w in all_windows)
    return all(prev[0] != nxt[0] or prev[2] <= nxt[1]
               for prev, nxt in zip(ordered, ordered[1:]))


def _queue_order(req: Request):
    return (-req.priority, req.submitted, req.earliest, req.robot)


def _window_key(window):
    return (window[0], window[1], window[2])


def _place(placed, earliest: float, duration: float):
    """Earliest feasible start at or after ``earliest`` and the intervals that set it.

    ``placed`` is a sorted list of non-overlapping ``(start, end, robot, source)``
    windows; the returned blockers are exactly the windows that pushed the start
    past ``earliest``, in the order they pushed.
    """
    start = earliest
    blockers = []
    for window in placed:
        if window[1] <= start:
            continue
        if window[0] >= start + duration:
            break
        start = window[1]
        blockers.append(window)
    return start, blockers


def _reason(first: Request, other: Request) -> str:
    if first.priority != other.priority:
        return f'priority {first.priority} beats {other.priority}'
    if first.submitted != other.submitted:
        return f'waited longer (submitted at {first.submitted:g} vs {other.submitted:g})'
    if first.earliest != other.earliest:
        return f'earlier desired start ({first.earliest:g} vs {other.earliest:g})'
    return 'identical terms; robot name breaks the tie'
