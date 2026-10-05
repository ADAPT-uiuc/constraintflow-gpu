"""
Keep the segmented flow under a device-memory budget by parking idle values on
the host.

flow_split cuts the flow into segments that the driver calls in order. Between
two calls the driver holds every value a later segment reads. When the values
held plus the next segment's own working set exceed the budget, the values
whose next reader is furthest away move to (pinned) host memory before the
call and come back right before the first segment that reads them again
(Belady's rule). Only the driver changes; segments and arithmetic do not.

Sizes come from the flow_shapes probe, so the plan is static.
"""

from constraintflow.compiler.optimizations import flow_split

# Smaller values free too little to be worth a round trip over PCIe.
MIN_OFFLOAD_BYTES = 64 * 1024 ** 2


def _gb(n):
    return '{:.1f} GB'.format(n / 1024 ** 3)


class Plan:
    def __init__(self, segments):
        self.evict = [[] for _ in segments]    # names off the device before call k
        self.fetch = [[] for _ in segments]    # names back on the device before call k
        # When the copies are issued: an eviction as soon as its value goes idle,
        # a fetch as early as the budget allows (never before its eviction).
        self.evict_issue = [[] for _ in segments]
        self.fetch_issue = [[] for _ in segments]
        self.held = [0 for _ in segments]      # estimated device bytes during call k
        self.peak = 0                          # estimated device peak with the plan
        self.peak_without = 0                  # same, without offloading
        self.moved_bytes = 0                   # device->host bytes per flow() call
        self.overflow = []                     # segments that do not fit even so

    def __bool__(self):
        return any(self.evict)

    def describe(self):
        return ('{} values offloaded, {} moved per run, estimated peak {} -> {}{}'.format(
            sum(map(len, self.evict)), _gb(self.moved_bytes), _gb(self.peak_without),
            _gb(self.peak),
            ', over budget in segments ' + ', '.join(map(str, self.overflow[:8]))
            if self.overflow else ''))


def _segment_peak(stmts, start, sized, root_of, keep):
    """Bytes a segment allocates at its peak, its inputs excluded."""
    reads = [[root_of.get(n) for n in flow_split._reads(s)] for s in stmts]
    last = {}
    for k, rs in enumerate(reads):
        for r in rs:
            last[r] = k
    alive, held, peak = {}, 0, 0
    for k in range(len(stmts)):
        v = sized.get(start + k)
        if v is not None:
            peak = max(peak, held + v.get('temporary_peak', 0))
            if v.get('alloc') and v['root'] not in alive:
                alive[v['root']] = v.get('storage_bytes', v['nbytes'])
                held += alive[v['root']]
                peak = max(peak, held)
        for r in reads[k]:
            if last.get(r) == k and r in alive and r not in keep:
                held -= alive.pop(r)
    return peak


def plan(stmts, segments, sized, budget):
    """Offload schedule for `segments` (consecutive slices of `stmts`).

    Segments that cannot fit even with everything else on the host are planned
    at the budget anyway: their estimate counts unfused temporaries, so relaxing
    the other segments to it raised the real peak (ResNet34: 67 -> 75 GiB).
    """
    result = Plan(segments)
    root_of, size = {}, {}
    params = set()
    for name, v in getattr(sized, 'inputs', {}).items():
        root_of[name] = v['root']
        size[v['root']] = v['storage_bytes']
        params.add(v['root'])
    for i, v in sized.items():
        root_of[v['name']] = v['root']
        size[v['root']] = v.get('storage_bytes', v['nbytes'])
    baseline = sum(size[r] for r in params)

    # Values crossing segment boundaries, and which segments read them.
    readers = {}
    crossing = {}
    for k, seg in enumerate(segments):
        for n in seg.live_in:
            r = root_of.get(n)
            if r is None or r in params:
                continue
            readers.setdefault(r, []).append(k)
            crossing.setdefault(r, set()).add(n)
        for n in seg.live_out:
            r = root_of.get(n)
            if r is not None and r not in params:
                crossing.setdefault(r, set()).add(n)
    # A root reachable through two names (a view and its base) stays put.
    movable = {r: next(iter(ns)) for r, ns in crossing.items() if len(ns) == 1}

    def next_read(r, k):
        for j in readers.get(r, ()):
            if j >= k:
                return j
        return None

    start, resident, host = 0, set(), set()
    for k, seg in enumerate(segments):
        keep = {root_of.get(n) for n in seg.live_out}
        work = _segment_peak(seg.stmts, start, sized, root_of, keep)
        start += len(seg.stmts)
        need = {root_of[n] for n in seg.live_in if n in root_of and root_of[n] not in params}
        for r in sorted(need & host):
            result.fetch[k].append(movable[r])
        host -= need
        resident |= need
        held = baseline + sum(size[r] for r in resident) + work
        result.peak_without = max(result.peak_without,
                                  held + sum(size[r] for r in host))
        while held > budget:
            victims = [r for r in resident - need
                       if r in movable and size[r] >= MIN_OFFLOAD_BYTES]
            if not victims:
                result.overflow.append('{} ({} stmts, inputs {}, working set {})'.format(
                    k, len(seg.stmts), _gb(sum(size[r] for r in need)), _gb(work)))
                break
            r = max(victims, key=lambda r: (next_read(r, k), size[r]))
            resident.discard(r)
            host.add(r)
            result.evict[k].append(movable[r])
            result.moved_bytes += size[r]
            held -= size[r]
        result.peak = max(result.peak, held)
        result.held[k] = held
        resident |= {r for r in keep if r is not None and r not in params}
        dead = {r for r in resident | host if next_read(r, k + 1) is None}
        resident -= dead
        host -= dead
    _schedule_copies(result, segments, root_of, {movable[r]: size[r] for r in movable}, budget)
    return result


def _schedule_copies(result, segments, root_of, nbytes, budget):
    """Issue points that overlap the copies with segments, within the plan's memory."""
    touched = {}                               # name -> segments reading or writing it
    for k, seg in enumerate(segments):
        for n in list(seg.live_in) + list(seg.live_out):
            touched.setdefault(n, []).append(k)
    evicted_at = {}
    for k, names in enumerate(result.evict):
        for n in names:
            # the call after the value's last use before k; nothing reads it after
            issue = 1 + max((j for j in touched.get(n, ()) if j < k), default=-1)
            result.evict_issue[max(issue, 0)].append(n)
            evicted_at.setdefault(n, []).append(k)
    for k, names in enumerate(result.fetch):
        for n in names:
            floor = max((e for e in evicted_at.get(n, ()) if e < k), default=0)
            j = k
            while j > floor and result.held[j - 1] + nbytes[n] <= budget:
                j -= 1
            for i in range(j, k):
                result.held[i] += nbytes[n]
            result.fetch_issue[j].append(n)
