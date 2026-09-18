"""
Shifting Bottleneck Heuristic (SBH) for the job shop scheduling problem.

Deterministic implementation: given jobs (routing + processing times per
operation), this computes, machine by machine, which machine is the current
"bottleneck" (largest achievable max-lateness on its one-machine subproblem)
and fixes that machine's job sequence, repeating until every machine has a
fixed sequence. No values are hand-computed or guessed -- every number here
comes from the graph/longest-path arithmetic below.
"""

from dataclasses import dataclass, field
from itertools import permutations


@dataclass
class Job:
    job_id: int
    routing: list          # machines in chronological order, e.g. [2, 3, 1]
    proc_times: list       # processing times aligned with routing order

    def machine_of_step(self, step_index):
        return self.routing[step_index]

    def proc_time_on_machine(self, machine):
        idx = self.routing.index(machine)
        return self.proc_times[idx]

    def step_index_of_machine(self, machine):
        return self.routing.index(machine)


SOURCE = ("U",)
SINK = ("V",)


def op_node(job_id, step_index):
    return ("op", job_id, step_index)


def build_graph(jobs, fixed_machine_sequences):
    """
    Builds the disjunctive graph's currently-fixed edges:
      - conjunctive arcs (job routing precedence), always present
      - disjunctive arcs already fixed for machines in fixed_machine_sequences
    Returns adjacency dict: node -> list of (successor, weight)
    weight on edge (X -> Y) = processing time of operation X
    (dummy SOURCE has 0-weight edges into first ops; last ops have edges into
    SINK weighted by their own processing time).
    """
    adj = {}

    def add_edge(u, v, w):
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, [])

    adj[SOURCE] = []
    adj[SINK] = []

    for job in jobs:
        n = len(job.routing)
        for k in range(n):
            node = op_node(job.job_id, k)
            adj.setdefault(node, [])
        # source -> first op
        add_edge(SOURCE, op_node(job.job_id, 0), 0)
        # conjunctive arcs
        for k in range(n - 1):
            dur_k = job.proc_times[k]
            add_edge(op_node(job.job_id, k), op_node(job.job_id, k + 1), dur_k)
        # last op -> sink
        last_dur = job.proc_times[n - 1]
        add_edge(op_node(job.job_id, n - 1), SINK, last_dur)

    # disjunctive arcs already fixed
    for machine, seq in fixed_machine_sequences.items():
        for a, b in zip(seq, seq[1:]):
            ja, ka = a
            job_a = next(j for j in jobs if j.job_id == ja)
            dur_a = job_a.proc_time_on_machine(machine)
            add_edge(op_node(ja, ka), op_node(*b), dur_a)

    return adj


def topo_order(adj):
    """Kahn's algorithm topological sort."""
    indeg = {u: 0 for u in adj}
    for u in adj:
        for v, _ in adj[u]:
            indeg[v] = indeg.get(v, 0) + 1
            indeg.setdefault(u, indeg.get(u, 0))
    from collections import deque
    q = deque([u for u in adj if indeg.get(u, 0) == 0])
    order = []
    indeg = dict(indeg)
    while q:
        u = q.popleft()
        order.append(u)
        for v, _ in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order


def longest_paths(adj):
    """
    Forward longest path from SOURCE (dist = earliest start time of each node)
    and backward longest path to SINK (rdist = longest path length from node
    to SINK, INCLUDING that node's own duration -- because edge weights are
    the source node's own duration).
    """
    order = topo_order(adj)
    dist = {u: 0 for u in adj}
    for u in order:
        for v, w in adj.get(u, []):
            if dist[u] + w > dist.get(v, 0):
                dist[v] = dist[u] + w

    # reverse graph for backward pass
    radj = {u: [] for u in adj}
    for u in adj:
        for v, w in adj[u]:
            radj[v].append((u, w))

    rdist = {u: 0 for u in adj}
    for u in reversed(order):
        for v, w in radj.get(u, []):
            # path v -> u -> ... -> SINK ; weight w = dur(v)
            if w + rdist.get(u, 0) > rdist.get(v, 0):
                rdist[v] = w + rdist[u]

    return dist, rdist


def compute_heads_tails(jobs, fixed_machine_sequences):
    adj = build_graph(jobs, fixed_machine_sequences)
    dist, rdist = longest_paths(adj)
    cmax = dist[SINK]

    heads = {}   # (job_id, machine) -> r_ij
    tails = {}   # (job_id, machine) -> tail after this op (successors only)
    deadlines = {}  # (job_id, machine) -> d_ij
    durations = {}

    for job in jobs:
        for k, machine in enumerate(job.routing):
            node = op_node(job.job_id, k)
            dur = job.proc_times[k]
            head = dist[node]
            tail_incl = rdist[node]
            tail_excl = tail_incl - dur
            heads[(job.job_id, machine)] = head
            tails[(job.job_id, machine)] = tail_excl
            deadlines[(job.job_id, machine)] = cmax - tail_excl
            durations[(job.job_id, machine)] = dur

    return {
        "cmax": cmax,
        "heads": heads,
        "tails": tails,
        "deadlines": deadlines,
        "durations": durations,
    }


def solve_one_machine_lmax(ops):
    """
    Exact solution of 1 || Lmax (single machine, release dates, no
    preemption) via brute-force enumeration of job orders.
    ops: list of dict(job_id=, r=, p=, d=)
    Returns (best_sequence_job_ids, best_Lmax, per_job_completion_and_lateness)
    """
    n = len(ops)
    best_seq = None
    best_lmax = None
    best_detail = None

    for perm in permutations(range(n)):
        t = 0
        lmax = float("-inf")
        detail = []
        for idx in perm:
            o = ops[idx]
            start = max(t, o["r"])
            finish = start + o["p"]
            lateness = finish - o["d"]
            t = finish
            lmax = max(lmax, lateness)
            detail.append({
                "job_id": o["job_id"], "start": start, "finish": finish,
                "lateness": lateness,
            })
        if best_lmax is None or lmax < best_lmax:
            best_lmax = lmax
            best_seq = [ops[idx]["job_id"] for idx in perm]
            best_detail = detail

    return best_seq, best_lmax, best_detail


def run_shifting_bottleneck(jobs):
    """
    Full multi-iteration shifting bottleneck heuristic.
    Returns a dict with the iteration log and final results.
    """
    all_machines = sorted({m for job in jobs for m in job.routing})
    fixed_sequences = {}   # machine -> [(job_id, step_index), ...] in fixed order
    iterations = []

    while len(fixed_sequences) < len(all_machines):
        state = compute_heads_tails(jobs, fixed_sequences)
        cmax = state["cmax"]

        candidates = {}
        for m in all_machines:
            if m in fixed_sequences:
                continue
            ops = []
            for job in jobs:
                if m in job.routing:
                    ops.append({
                        "job_id": job.job_id,
                        "r": state["heads"][(job.job_id, m)],
                        "p": state["durations"][(job.job_id, m)],
                        "d": state["deadlines"][(job.job_id, m)],
                    })
            seq, lmax, detail = solve_one_machine_lmax(ops)
            candidates[m] = {"ops": ops, "seq": seq, "lmax": lmax, "detail": detail}

        bottleneck_machine = max(candidates, key=lambda m: candidates[m]["lmax"])
        chosen = candidates[bottleneck_machine]

        iterations.append({
            "cmax_before": cmax,
            "candidates": candidates,
            "chosen_machine": bottleneck_machine,
        })

        seq_nodes = []
        for jid in chosen["seq"]:
            job = next(j for j in jobs if j.job_id == jid)
            seq_nodes.append((jid, job.step_index_of_machine(bottleneck_machine)))
        fixed_sequences[bottleneck_machine] = seq_nodes

    final_state = compute_heads_tails(jobs, fixed_sequences)
    final_cmax = final_state["cmax"]

    per_machine_schedule = {}
    max_lateness_overall = float("-inf")
    for it in iterations:
        m = it["chosen_machine"]
        chosen = it["candidates"][m]
        per_machine_schedule[m] = chosen["detail"]
        max_lateness_overall = max(max_lateness_overall, chosen["lmax"])

    return {
        "all_machines": all_machines,
        "fixed_sequences": fixed_sequences,
        "iterations": iterations,
        "final_cmax": final_cmax,
        "final_state": final_state,
        "per_machine_schedule": per_machine_schedule,
        "max_lateness_overall": max_lateness_overall,
    }
