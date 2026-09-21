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

import matplotlib.pyplot as plt


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
    Exact solution of 1|rj|Lmax (single machine, release dates, no
    preemption) via brute-force enumeration of ALL job orders -- this is
    not a greedy EDD/deadline sort, so it does not silently assume a tie
    in deadlines (dij) means a tie in the optimal schedule: with release
    dates, order still changes completion times, so two jobs sharing a
    deadline can still have only one truly-optimal relative order.
    ops: list of dict(job_id=, r=, p=, d=)
    Returns (best_sequence_job_ids, best_Lmax, per_job_completion_and_lateness, all_optimal)
    where all_optimal is a list of {"seq": [...], "detail": [...]} for
    EVERY permutation that genuinely ties the best achievable Lmax (usually
    just one entry, but more when a true tie exists).
    """
    n = len(ops)
    best_lmax = None
    all_optimal = []

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
        seq = [ops[idx]["job_id"] for idx in perm]
        if best_lmax is None or lmax < best_lmax:
            best_lmax = lmax
            all_optimal = [{"seq": seq, "detail": detail}]
        elif lmax == best_lmax:
            all_optimal.append({"seq": seq, "detail": detail})

    best_seq = all_optimal[0]["seq"]
    best_detail = all_optimal[0]["detail"]
    return best_seq, best_lmax, best_detail, all_optimal


# Fixed grid spacing (in axes units) -- every node sits on this grid, never
# on a data-dependent coordinate, so nodes can't collide or crowd together
# regardless of processing-time values.
GRID_X_GAP = 3.0    # horizontal distance between successive routing steps
GRID_Y_GAP = 2.0     # vertical distance between job rows
NODE_SIZE = 1900


def draw_disjunctive_graph(jobs, fixed_machine_sequences, title=""):
    """
    Draws the disjunctive graph in its current state:
      - solid directed arcs, labeled with weight: conjunctive arcs (always)
        plus any disjunctive arcs already fixed for a machine
      - dashed undirected lines, one color per machine: disjunctive pairs for
        machines NOT yet in fixed_machine_sequences

    Layout is a fixed grid, NOT time/head-based: each job is its own row
    (spaced GRID_Y_GAP apart) and each routing step is its own column
    (spaced GRID_X_GAP apart, left to right in that job's routing order).
    This keeps nodes evenly spaced and arrows readable no matter what the
    processing times are. Returns a matplotlib Figure.
    """
    adj = build_graph(jobs, fixed_machine_sequences)
    jobs_by_id = {j.job_id: j for j in jobs}

    n_rows = max(len(jobs), 1)
    max_steps = max((len(j.routing) for j in jobs), default=1)
    row_y = {j.job_id: (n_rows - 1 - i) * GRID_Y_GAP for i, j in enumerate(jobs)}
    mid_y = (n_rows - 1) * GRID_Y_GAP / 2

    def pos(node):
        if node == SOURCE:
            return (-1 * GRID_X_GAP, mid_y)
        if node == SINK:
            return (max_steps * GRID_X_GAP, mid_y)
        _, jid, k = node
        return (k * GRID_X_GAP, row_y[jid])

    fig, ax = plt.subplots(figsize=(3 + GRID_X_GAP * (max_steps + 2),
                                     1.4 * n_rows + 2))

    # conjunctive arcs + any already-fixed disjunctive arcs: solid, directed
    for u, edges in adj.items():
        for v, w in edges:
            x1, y1 = pos(u)
            x2, y2 = pos(v)
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4,
                                 shrinkA=26, shrinkB=26,
                                 connectionstyle="arc3,rad=0.12" if y1 != y2 else "arc3,rad=0"),
            )
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            offset = 0.18 if y1 == y2 else 0.0
            ax.text(mx, my + GRID_Y_GAP * 0.12 + offset, str(w), fontsize=9,
                    ha="center", va="bottom", color="#222",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))

    # disjunctive pairs not yet resolved: dashed, undirected, one color/machine
    all_machines = sorted({m for j in jobs for m in j.routing})
    colors = plt.cm.Set1.colors
    for mi, m in enumerate(all_machines):
        if m in fixed_machine_sequences:
            continue
        ops_on_m = [op_node(j.job_id, j.step_index_of_machine(m))
                    for j in jobs if m in j.routing]
        color = colors[mi % len(colors)]
        for i in range(len(ops_on_m)):
            for k in range(i + 1, len(ops_on_m)):
                x1, y1 = pos(ops_on_m[i])
                x2, y2 = pos(ops_on_m[k])
                ax.plot([x1, x2], [y1, y2], linestyle="--", color=color,
                        lw=1.4, alpha=0.85, zorder=1)
        # legend entry
        ax.plot([], [], linestyle="--", color=color, label=f"Machine {m} (unresolved)")

    # node markers on top
    for node in adj:
        x, y = pos(node)
        if node == SOURCE:
            label = "U"
        elif node == SINK:
            label = "V"
        else:
            _, jid, k = node
            job = jobs_by_id[jid]
            m = job.machine_of_step(k)
            label = f"J{jid}\nM{m}"
        ax.scatter([x], [y], s=NODE_SIZE, color="#f4f4f4", edgecolor="black",
                   linewidth=1.4, zorder=3)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=4)

    if any(m not in fixed_machine_sequences for m in all_machines):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                   ncol=min(len(all_machines), 4), fontsize=9, frameon=False)

    ax.set_ylim(-GRID_Y_GAP, n_rows * GRID_Y_GAP)
    ax.set_xlim(-2.2 * GRID_X_GAP, (max_steps + 1.2) * GRID_X_GAP)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=14)
    fig.tight_layout()
    return fig


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
            seq, lmax, detail, all_optimal = solve_one_machine_lmax(ops)
            candidates[m] = {
                "ops": ops, "seq": seq, "lmax": lmax, "detail": detail,
                "all_optimal": all_optimal,  # every sequence tying the best Lmax
            }

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
        iterations[-1]["fixed_sequences_after"] = dict(fixed_sequences)

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
