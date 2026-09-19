"""Batch reservation planning with starvation-aware conflict resolution.

The one-at-a-time :class:`~aegisrover.mission.allocation.ReservationBook`
workflow forces callers to discover conflicts by trial and error: reserve,
catch :class:`~aegisrover.mission.allocation.Conflict`, reorder, retry. This
module accepts a whole batch of requests at once and returns a feasible
:class:`Plan` instead. Every request either gets a concrete window on its
resource or an explicit rejection saying what blocks it. When several robots
want the same resource at the same time, the plan records *who goes first and
why* as a :class:`Decision`, and reports per-robot waiting times so a starved
robot is visible instead of silently losing every retry.

Ordering rule (first difference wins, recorded as the decision reason):

1. higher ``priority`` first — mission importance;
2. longer wait (``now - enqueued_at``) first — the hungriest robot is
   favoured, which is the anti-starvation rule;
3. earlier ``deadline`` first;
4. lower robot id — deterministic tie-break.

Placement is greedy: in that order each request takes the earliest free
window on its resource that starts no earlier than ``earliest`` and ends no
later than ``deadline``. Windows are half-open, matching
:class:`~aegisrover.mission.allocation.Reservation`, so a follower may start
at the exact instant its predecessor releases the resource.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable

from aegisrover.mission.allocation import Reservation

__all__ = ('Request', 'PlanEntry', 'Decision', 'Rejection', 'Plan', 'plan_batch')


@dataclass(frozen=True)
class Request:
    """One robot asking for one resource window in a batch plan."""

    robot: str
    resource: str
    earliest: float
    duration: float
    deadline: float = inf
    priority: int = 0
    enqueued_at: float = 0.0

    def __post_init__(self):
        if not self.robot or not self.resource:
            raise ValueError('robot and resource are required')
        if self.duration <= 0:
            raise ValueError('duration must be positive')
        if self.deadline < self.earliest:
            raise ValueError('deadline must not precede earliest')

    @property
    def desired_end(self) -> float:
        return self.earliest + self.duration

    def contends_with(self, other: 'Request') -> bool:
        """True when both requests want overlapping windows on one resource."""
        return (self.resource == other.resource and self.robot != other.robot
                and self.earliest < other.desired_end and other.earliest < self.desired_end)


@dataclass(frozen=True)
class PlanEntry:
    """A scheduled window: ``request.robot`` holds the resource over [start, end)."""

    request: Request
    start: float
    end: float
    behind: tuple[str, ...]  # holders whose windows pushed this one past ``earliest``

    @property
    def wait(self) -> float:
        """Delay past the earliest possible start, caused by contention."""
        return self.start - self.request.earliest

    def reservation(self) -> Reservation:
        return Reservation(self.request.resource, self.request.robot, self.start,
                           self.end, priority=self.request.priority)


@dataclass(frozen=True)
class Decision:
    """Why one robot goes before another on a contended resource."""

    resource: str
    first: str
    then: str
    rule: str  # 'priority' | 'starvation' | 'deadline' | 'id'
    detail: str


@dataclass(frozen=True)
class Rejection:
    """A request no feasible window could be found for."""

    request: Request
    reason: str
    blockers: tuple[str, ...]  # holders occupying the resource over the feasible range


@dataclass(frozen=True)
class Plan:
    """The outcome of :func:`plan_batch`."""

    now: float
    entries: tuple[PlanEntry, ...]
    decisions: tuple[Decision, ...]
    rejections: tuple[Rejection, ...]

    def entry_for(self, robot: str) -> PlanEntry | None:
        return next((e for e in self.entries if e.request.robot == robot), None)

    def wait_times(self) -> dict[str, float]:
        """Robot -> seconds from ``now`` until its planned window starts."""
        return {e.request.robot: max(0.0, e.start - self.now) for e in self.entries}

    def starvation(self) -> tuple[tuple[str, float], ...]:
        """(robot, seconds already waiting) pairs, hungriest first.

        Covers every robot in the batch, scheduled or not — a rejected robot
        is usually the one starvation matters most for.
        """
        robots = [e.request for e in self.entries] + [r.request for r in self.rejections]
        ages = {r.robot: self.now - r.enqueued_at for r in robots}
        return tuple(sorted(ages.items(), key=lambda kv: (-kv[1], kv[0])))

    def summary(self) -> str:
        lines = [f'plan @now={self.now:g}: {len(self.entries)} scheduled, '
                 f'{len(self.rejections)} rejected']
        for entry in self.entries:
            req = entry.request
            behind = f', behind {", ".join(entry.behind)}' if entry.behind else ''
            lines.append(f'  {req.resource}: {req.robot} {entry.start:g}-{entry.end:g}'
                         f' (wait {entry.wait:g}s{behind})')
        for decision in self.decisions:
            lines.append(f'  decision {decision.resource}: {decision.first} before '
                         f'{decision.then} ({decision.rule}: {decision.detail})')
        for rejection in self.rejections:
            lines.append(f'  rejected {rejection.request.robot}@{rejection.request.resource}:'
                         f' {rejection.reason}')
        if self.entries or self.rejections:
            ranking = ', '.join(f'{robot} {age:g}s' for robot, age in self.starvation())
            lines.append(f'  starvation: {ranking}')
        return '\n'.join(lines)


def plan_batch(requests: Iterable[Request], existing: Iterable[Reservation] = (),
               now: float = 0.0) -> Plan:
    """Schedule a batch of requests against already-committed reservations.

    ``existing`` reservations (e.g. ``ReservationBook.items()``) are fixed
    obstacles; batch requests are placed around them. The result never raises
    on contention: conflicts become :class:`Decision` entries and impossible
    requests become :class:`Rejection` entries.
    """
    ordered = sorted(requests, key=lambda r: _order_key(r, now))
    decisions = _decisions(ordered, now)

    busy: dict[str, list[tuple[float, float, str]]] = {}
    for item in existing:
        busy.setdefault(item.resource, []).append((item.start, item.end, item.holder))

    entries: list[PlanEntry] = []
    rejections: list[Rejection] = []
    for request in ordered:
        intervals = busy.setdefault(request.resource, [])
        placed = _earliest_slot(request, intervals)
        if placed is None:
            blockers = tuple(sorted({holder for _, _, holder in intervals}))
            rejections.append(Rejection(
                request, f'no free window of {request.duration:g}s before '
                         f'deadline {request.deadline:g}', blockers))
            continue
        start, behind = placed
        end = start + request.duration
        entries.append(PlanEntry(request, start, end, behind))
        intervals.append((start, end, request.robot))
    return Plan(now, tuple(entries), decisions, tuple(rejections))


def _order_key(request: Request, now: float):
    return (-request.priority, -(now - request.enqueued_at), request.deadline, request.robot)


def _decisions(ordered: list[Request], now: float) -> tuple[Decision, ...]:
    out: list[Decision] = []
    for i, first in enumerate(ordered):
        for second in ordered[i + 1:]:
            if first.contends_with(second):
                out.append(_explain(first, second, now))
    return tuple(out)


def _explain(first: Request, second: Request, now: float) -> Decision:
    """Describe why ``first`` (earlier in the order) outranks ``second``."""
    if first.priority != second.priority:
        rule, detail = 'priority', f'priority {first.priority} beats {second.priority}'
    else:
        first_age, second_age = now - first.enqueued_at, now - second.enqueued_at
        if first_age != second_age:
            rule = 'starvation'
            detail = f'{first.robot} has waited {first_age:g}s vs {second.robot} {second_age:g}s'
        elif first.deadline != second.deadline:
            rule, detail = 'deadline', (f'deadline {first.deadline:g} before '
                                        f'{second.deadline:g}')
        else:
            rule, detail = 'id', f'tie broken by robot id {first.robot!r}'
    return Decision(first.resource, first.robot, second.robot, rule, detail)


def _earliest_slot(request: Request, intervals: list[tuple[float, float, str]]):
    """Earliest start >= request.earliest fitting before the deadline.

    Returns ``(start, behind)`` where ``behind`` names the holders whose
    windows pushed the start past ``request.earliest``, or ``None`` when no
    window of ``request.duration`` ends by ``request.deadline``.
    """
    start = request.earliest
    behind: list[str] = []
    for busy_start, busy_end, holder in sorted(intervals):
        if busy_end <= start:
            continue
        if busy_start >= start + request.duration:
            break
        start = busy_end
        if holder != request.robot and holder not in behind:
            behind.append(holder)
    if start + request.duration > request.deadline:
        return None
    return start, tuple(behind)
