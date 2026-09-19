"""Storage, audit and mission-domain acceptance tests for the v1.1 platform."""
import pytest

from aegisrover.mission.allocation import Conflict, Deadlock, Reservation, ReservationBook
from aegisrover.mission.batch import Request, plan_batch
from aegisrover.mission.lifecycle import InvalidTransition, MissionService
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import NotFound, Repository, VersionConflict


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        self.now += 0.5
        return self.now


@pytest.fixture()
def repo():
    r = Repository(':memory:', clock=FakeClock())
    yield r
    r.close()


def test_repository_versions_and_history(repo):
    first = repo.put('maps', 'yard', {'rev': 1})
    assert first.version == 1
    second = repo.put('maps', 'yard', {'rev': 2}, expected=1)
    assert (second.version, second.etag) != (first.version, first.etag)
    with pytest.raises(VersionConflict):
        repo.put('maps', 'yard', {'rev': 3}, expected=1)
    assert [r.version for r in repo.history('maps', 'yard')] == [1, 2]
    assert repo.get('maps', 'yard').payload == {'rev': 2}


def test_repository_paging_and_tombstones(repo):
    for i in range(7):
        repo.put('missions', f'm{i:02d}', {'i': i})
    page = repo.page('missions', limit=3)
    assert [r.key for r in page.items] == ['m00', 'm01', 'm02']
    assert page.has_more and page.cursor
    page2 = repo.page('missions', limit=3, cursor=page.cursor)
    assert [r.key for r in page2.items] == ['m03', 'm04', 'm05']
    repo.delete('missions', 'm04', expected=1)
    with pytest.raises(NotFound):
        repo.get('missions', 'm04')
    assert repo.get('missions', 'm04', include_deleted=True).deleted
    assert repo.count('missions') == 6


def test_audit_chain_detects_tampering(repo):
    audit = AuditLog(repo, clock=FakeClock())
    audit.append('operator', 'mission.create', 'm1', {'priority': 1})
    audit.append('operator', 'mission.queue', 'm1')
    audit.append('robot-7', 'mission.start', 'm1')
    assert audit.verify() == ()
    assert [e.seq for e in audit.entries()] == [1, 2, 3]
    audit.tamper(2, action='mission.cancel')
    breaks = audit.verify()
    assert breaks and breaks[0].seq == 2
    assert any(b.reason == 'entry digest mismatch' for b in breaks)


def test_audit_chain_detects_reordering(repo):
    audit = AuditLog(repo, clock=FakeClock())
    for name in ('a', 'b', 'c'):
        audit.append('operator', 'note', name)
    record = repo.get('audit', '000000000002')
    payload = dict(record.payload)
    payload['previous_digest'] = '0' * 64
    repo.put('audit', '000000000002', payload)
    reasons = {b.reason for b in audit.verify()}
    assert 'previous digest mismatch' in reasons


def test_mission_lifecycle_is_guarded_and_idempotent(repo):
    service = MissionService(repo, clock=FakeClock())
    mission = service.create('m1', [(0, 0), (5, 0)], priority=2, actor='planner')
    assert mission.state == 'draft'
    assert service.command('m1', 'queue')['applied'] is True
    assert service.command('m1', 'queue')['applied'] is False
    with pytest.raises(InvalidTransition):
        service.command('m1', 'complete')
    service.command('m1', 'assign', assignee='robot-1')
    service.command('m1', 'start')
    service.command('m1', 'complete')
    with pytest.raises(InvalidTransition):
        service.command('m1', 'start')
    assert service.get('m1').state == 'completed'


def test_mission_conflicting_commands_are_serialised(repo):
    service = MissionService(repo, clock=FakeClock())
    service.create('m1', [(0, 0)], actor='planner')
    service.command('m1', 'queue')
    with pytest.raises(VersionConflict):
        service.command('m1', 'assign', assignee='robot-1', expected_revision=99)
    history = service.get('m1').history
    assert [h['command'] for h in history] == ['create', 'queue']


def test_mission_dispatch_respects_priority_and_capabilities(repo):
    service = MissionService(repo, clock=FakeClock())
    service.create('low', [(0, 0)], priority=0, required_capabilities=['lidar'])
    service.create('high', [(1, 1)], priority=9, required_capabilities=['lidar', 'arm'])
    for name in ('low', 'high'):
        service.command(name, 'queue')
    # A robot that cannot satisfy the highest-priority mission must not silently
    # take it; it gets the best mission it is actually able to run.
    first = service.claim_next('robot-1', ['lidar'])
    assert first.mission_id == 'low' and first.state == 'assigned'
    second = service.claim_next('robot-2', ['lidar', 'arm'])
    assert second.mission_id == 'high'
    assert service.claim_next('robot-3', ['lidar', 'arm']) is None


def test_reservations_are_half_open():
    book = ReservationBook()
    book.reserve(Reservation('door-a', 'robot-1', 10.0, 12.0))
    book.reserve(Reservation('door-a', 'robot-2', 12.0, 14.0))
    with pytest.raises(Conflict):
        book.reserve(Reservation('door-a', 'robot-3', 11.5, 12.5))
    assert book.utilization('door-a', 10.0, 14.0) == pytest.approx(1.0)
    assert book.free_windows('door-a', 9.0, 15.0) == [(9.0, 10.0), (14.0, 15.0)]


def test_reservation_preemption_only_when_lower_priority_has_not_started():
    book = ReservationBook()
    book.reserve(Reservation('dock', 'robot-1', 10.0, 12.0, priority=1))
    book.reserve(Reservation('dock', 'robot-2', 10.5, 12.5, priority=5), preempt=True, now=9.0)
    assert {r.holder for r in book.usage('dock')} == {'robot-2'}
    with pytest.raises(Conflict):
        # robot-2 already started at 10.5, so it may not be evicted implicitly.
        book.reserve(Reservation('dock', 'robot-3', 11.0, 13.0, priority=9), preempt=True, now=10.6)
    assert {r.holder for r in book.usage('dock')} == {'robot-2'}
    book.release('robot-2')
    book.reserve(Reservation('dock', 'robot-3', 11.0, 13.0, priority=9))
    assert {r.holder for r in book.usage('dock')} == {'robot-3'}


def test_wait_for_graph_reports_circular_wait():
    book = ReservationBook()
    book.reserve(Reservation('door-a', 'robot-1', 0.0, 5.0))
    book.reserve(Reservation('door-b', 'robot-2', 0.0, 5.0))
    book.wait_for('robot-1', 'robot-2')
    book.wait_for('robot-2', 'robot-1')
    cycle = book.detect_deadlock()
    assert cycle is not None and cycle[0] == cycle[-1] and set(cycle) == {'robot-1', 'robot-2'}
    with pytest.raises(Deadlock):
        book.require_no_deadlock()
    book.clear_wait('robot-2')
    assert book.detect_deadlock() is None


def test_batch_plan_orders_contended_resource_and_reports_waits():
    plan = plan_batch([
        Request('robot-1', 'door-a', earliest=10.0, duration=2.0, enqueued_at=9.0),
        Request('robot-2', 'door-a', earliest=10.0, duration=2.0, enqueued_at=5.0),
        Request('robot-3', 'dock', earliest=10.0, duration=1.0, enqueued_at=8.0),
    ], now=10.0)
    # robot-2 has been waiting longer, so it takes the door first; robot-1
    # starts at the exact instant robot-2 releases it (half-open windows).
    assert (plan.entry_for('robot-2').start, plan.entry_for('robot-2').end) == (10.0, 12.0)
    assert (plan.entry_for('robot-1').start, plan.entry_for('robot-1').end) == (12.0, 14.0)
    assert plan.entry_for('robot-1').wait == 2.0
    assert plan.entry_for('robot-1').behind == ('robot-2',)
    assert plan.entry_for('robot-3').wait == 0.0
    decision = plan.decisions[0]
    assert (decision.resource, decision.first, decision.then, decision.rule) == (
        'door-a', 'robot-2', 'robot-1', 'starvation')
    assert plan.wait_times() == {'robot-1': 2.0, 'robot-2': 0.0, 'robot-3': 0.0}
    assert plan.starvation()[0] == ('robot-2', 5.0)
    assert not plan.rejections
    assert 'robot-2 before robot-1' in plan.summary()
    # Deterministic: the same batch always yields the same plan.
    again = plan_batch([
        Request('robot-1', 'door-a', earliest=10.0, duration=2.0, enqueued_at=9.0),
        Request('robot-2', 'door-a', earliest=10.0, duration=2.0, enqueued_at=5.0),
        Request('robot-3', 'dock', earliest=10.0, duration=1.0, enqueued_at=8.0),
    ], now=10.0)
    assert again == plan


def test_batch_plan_decision_rule_precedence():
    # Priority outranks even a much longer wait.
    plan = plan_batch([
        Request('robot-1', 'door-a', 0.0, 1.0, priority=1, enqueued_at=-100.0),
        Request('robot-2', 'door-a', 0.0, 1.0, priority=5, enqueued_at=0.0),
    ], now=0.0)
    assert plan.decisions[0].rule == 'priority'
    assert plan.decisions[0].first == 'robot-2'
    # Equal priority and wait: earliest deadline goes first.
    plan = plan_batch([
        Request('robot-1', 'door-a', 0.0, 1.0, deadline=50.0),
        Request('robot-2', 'door-a', 0.0, 1.0, deadline=10.0),
    ], now=0.0)
    assert plan.decisions[0].rule == 'deadline'
    assert plan.decisions[0].first == 'robot-2'
    # A full tie is broken deterministically by robot id.
    plan = plan_batch([
        Request('robot-b', 'door-a', 0.0, 1.0),
        Request('robot-a', 'door-a', 0.0, 1.0),
    ])
    assert plan.decisions[0].rule == 'id'
    assert plan.decisions[0].first == 'robot-a'


def test_batch_plan_rejects_impossible_request_without_failing_batch():
    plan = plan_batch([
        Request('robot-1', 'door-a', earliest=10.0, duration=5.0, deadline=20.0),
        Request('robot-2', 'door-a', earliest=20.0, duration=5.0),
    ], existing=[Reservation('door-a', 'robot-9', 10.0, 20.0)], now=10.0)
    # robot-1 cannot fit before its deadline, but that must not sink robot-2.
    assert [r.request.robot for r in plan.rejections] == ['robot-1']
    rejection = plan.rejections[0]
    assert rejection.blockers == ('robot-9',)
    assert 'deadline' in rejection.reason
    assert (plan.entry_for('robot-2').start, plan.entry_for('robot-2').end) == (20.0, 25.0)
    # A rejected robot still shows up in the starvation ranking.
    assert 'robot-1' in dict(plan.starvation())


def test_reservation_book_plans_and_commits_batch():
    book = ReservationBook()
    book.reserve(Reservation('door-a', 'robot-9', 10.0, 12.0))
    plan = book.plan([
        Request('robot-1', 'door-a', earliest=10.0, duration=2.0),
        Request('robot-2', 'door-a', earliest=10.0, duration=2.0, priority=5),
    ], now=10.0)
    assert plan.decisions[0].rule == 'priority'
    scheduled = sorted(book.usage('door-a'), key=lambda r: r.start)
    assert [(r.holder, r.start, r.end) for r in scheduled] == [
        ('robot-9', 10.0, 12.0), ('robot-2', 12.0, 14.0), ('robot-1', 14.0, 16.0)]
    # A preview plans against the book but leaves it untouched.
    preview = book.plan([Request('robot-3', 'door-a', earliest=10.0, duration=1.0)],
                        now=10.0, commit=False)
    assert preview.entry_for('robot-3').start == 16.0
    assert {r.holder for r in book.usage('door-a')} == {'robot-1', 'robot-2', 'robot-9'}
