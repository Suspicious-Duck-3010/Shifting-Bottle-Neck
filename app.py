import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sbh import Job, run_shifting_bottleneck, compute_heads_tails

st.set_page_config(page_title="Job Shop - Shifting Bottleneck Heuristic", layout="wide")

if "jobs" not in st.session_state:
    st.session_state.jobs = {}   # job_id -> Job

title_col, reset_col = st.columns([5, 1])
with title_col:
    st.title("Job Shop Scheduling - Shifting Bottleneck Heuristic")
with reset_col:
    st.write("")
    if st.button("Reset", help="Clear all saved jobs and results to start a new problem"):
        st.session_state.jobs = {}
        st.session_state.pop("result", None)
        st.session_state.pop("initial_state", None)
        st.rerun()

st.markdown(
    "Enter each job's routing and processing times yourself. All scheduling "
    "math (heads/tails/deadlines, one-machine sequencing, makespan, lateness) "
    "is computed deterministically below -- nothing is filled in for you."
)

# ---------------------------------------------------------------- data entry
st.header("1. Enter job data")

col_a, col_b = st.columns([1, 2])

with col_a:
    n_jobs = st.number_input("Number of jobs", min_value=1, max_value=12, value=3, step=1)
    job_options = [f"Job {i}" for i in range(1, n_jobs + 1)]
    selected_label = st.selectbox("Select job to edit", job_options)
    selected_id = int(selected_label.split()[1])

    existing = st.session_state.jobs.get(selected_id)
    default_routing = ",".join(str(m) for m in existing.routing) if existing else ""
    default_proc = ",".join(str(p) for p in existing.proc_times) if existing else ""

    routing_str = st.text_input(
        "Routing (machine numbers, chronological order, comma-separated)",
        value=default_routing, placeholder="e.g. 2,3,1",
    )
    proc_str = st.text_input(
        "Processing times (same chronological order as routing, comma-separated)",
        value=default_proc, placeholder="e.g. 6,9,7",
    )

    if st.button("Save job"):
        try:
            routing = [int(x.strip()) for x in routing_str.split(",") if x.strip() != ""]
            proc_times = [int(x.strip()) for x in proc_str.split(",") if x.strip() != ""]
            if len(routing) == 0 or len(routing) != len(proc_times):
                st.error("Routing and processing times must be the same, non-zero length.")
            elif len(set(routing)) != len(routing):
                st.error("A job cannot visit the same machine twice in this model.")
            else:
                st.session_state.jobs[selected_id] = Job(selected_id, routing, proc_times)
                st.success(f"Saved Job {selected_id}: routing {routing}, times {proc_times}")
        except ValueError:
            st.error("Use whole numbers separated by commas.")

with col_b:
    st.subheader("Routing check (visual)")
    job = st.session_state.jobs.get(selected_id)
    if job:
        fig, ax = plt.subplots(figsize=(6, 1.4))
        xs = list(range(len(job.routing)))
        ax.plot(xs, [0] * len(xs), color="#888", zorder=1)
        for x, m, p in zip(xs, job.routing, job.proc_times):
            ax.scatter([x], [0], s=1200, color="#4C78A8", zorder=2)
            ax.text(x, 0, f"M{m}", ha="center", va="center", color="white",
                     fontsize=11, fontweight="bold", zorder=3)
            ax.text(x, -0.35, f"p={p}", ha="center", va="center", fontsize=9)
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_ylim(-0.8, 0.5)
        ax.axis("off")
        ax.set_title(f"Job {selected_id} routing: " + " -> ".join(f"M{m}" for m in job.routing))
        st.pyplot(fig)
    else:
        st.info("Enter and save this job's routing to see it drawn here.")

st.divider()

st.subheader("All saved jobs")
if st.session_state.jobs:
    rows = []
    for jid in sorted(st.session_state.jobs):
        j = st.session_state.jobs[jid]
        rows.append({
            "Job": jid,
            "Routing": " -> ".join(f"M{m}" for m in j.routing),
            "Processing times (in routing order)": ", ".join(str(p) for p in j.proc_times),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
else:
    st.info("No jobs saved yet.")

jobs_list = [st.session_state.jobs[jid] for jid in sorted(st.session_state.jobs)]
all_machines_present = sorted({m for j in jobs_list for m in j.routing})

# ------------------------------------------------------------- run + output
st.header("2. Run the heuristic")

run = st.button("Simulate", type="primary", disabled=(len(jobs_list) != n_jobs or n_jobs == 0))
if len(jobs_list) != n_jobs:
    st.warning(f"Save all {n_jobs} job(s) before simulating ({len(jobs_list)} saved so far).")

if run:
    st.session_state.result = run_shifting_bottleneck(jobs_list)
    st.session_state.initial_state = compute_heads_tails(jobs_list, {})

result = st.session_state.get("result")
initial_state = st.session_state.get("initial_state")

if result:
    st.divider()
    st.header("3. Matrices")

    st.subheader("Routing matrix (chronological machine order per job)")
    st.dataframe(pd.DataFrame(
        {f"Job {j.job_id}": j.routing for j in jobs_list}
    ).T.rename(columns=lambda c: f"Step {c+1}"), use_container_width=True)

    st.subheader("Processing time matrix (Pij, in routing order)")
    st.dataframe(pd.DataFrame(
        {f"Job {j.job_id}": j.proc_times for j in jobs_list}
    ).T.rename(columns=lambda c: f"Step {c+1}"), use_container_width=True)

    st.subheader("Per-machine Pij / rij / dij (from the INITIAL graph, before any disjunctive arcs)")
    for m in all_machines_present:
        cols = [j.job_id for j in jobs_list if m in j.routing]
        data = {"": ["Pij", "rij", "dij"]}
        for jid in cols:
            data[f"Job {jid}"] = [
                initial_state["durations"][(jid, m)],
                initial_state["heads"][(jid, m)],
                initial_state["deadlines"][(jid, m)],
            ]
        st.markdown(f"**Machine {m}**")
        st.dataframe(pd.DataFrame(data).set_index(""), use_container_width=True)

    st.caption(f"Cmax before adding any disjunctive arcs: **{initial_state['cmax']}**")

    st.divider()
    st.header("4. Shifting bottleneck iterations")

    for i, it in enumerate(result["iterations"], start=1):
        st.subheader(f"Iteration {i}")
        st.write(f"Cmax going into this iteration: **{it['cmax_before']}**")
        for m, cand in it["candidates"].items():
            marker = "  <- CHOSEN (bottleneck)" if m == it["chosen_machine"] else ""
            st.markdown(f"Machine {m}: best achievable Lmax = **{cand['lmax']}**, "
                        f"EDD-optimal sequence = {cand['seq']}{marker}")
            detail_df = pd.DataFrame(cand["detail"]).rename(columns={
                "job_id": "Job", "start": "Start", "finish": "Completion", "lateness": "Lateness",
            })
            st.dataframe(detail_df, hide_index=True, use_container_width=True)

    st.divider()
    st.header("5. Final schedule per machine (Jobs / Completion Times / Lateness)")
    for m in result["all_machines"]:
        detail = result["per_machine_schedule"][m]
        st.markdown(f"**Machine {m}** - fixed sequence: {[d['job_id'] for d in detail]}")
        df = pd.DataFrame(detail).rename(columns={
            "job_id": "Job", "start": "Start", "finish": "Completion", "lateness": "Lateness",
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.divider()
    st.header("6. Results")
    c1, c2 = st.columns(2)
    c1.metric("Total makespan (Cmax)", result["final_cmax"])
    c2.metric("Maximum lateness", result["max_lateness_overall"])

    st.divider()
    st.header("7. Flow diagrams")

    def gantt(ax, title, machine_order_fn):
        colors = plt.cm.tab10.colors
        job_color = {j.job_id: colors[i % 10] for i, j in enumerate(jobs_list)}
        y_ticks, y_labels = [], []
        for row, m in enumerate(result["all_machines"]):
            y_ticks.append(row)
            y_labels.append(f"M{m}")
            for d in machine_order_fn(m):
                ax.barh(row, d["finish"] - d["start"], left=d["start"], height=0.6,
                        color=job_color[d["job_id"]], edgecolor="black")
                ax.text(d["start"] + (d["finish"] - d["start"]) / 2, row, f"J{d['job_id']}",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time")
        ax.set_title(title)
        ax.invert_yaxis()

    left, right = st.columns(2)
    with left:
        st.subheader("Initial routing flow (as entered, no machine-conflict resolution)")
        fig1, ax1 = plt.subplots(figsize=(6, 2 + 0.4 * len(all_machines_present)))
        naive = {m: [] for m in all_machines_present}
        for j in jobs_list:
            t = 0
            for m, p in zip(j.routing, j.proc_times):
                naive[m].append({"job_id": j.job_id, "start": t, "finish": t + p})
                t += p
        gantt(ax1, "Initial (per-job routing only)", lambda m: naive[m])
        st.pyplot(fig1)
        st.caption("Illustrative only: each job's own operations back-to-back, ignoring that machines are shared.")

    with right:
        st.subheader("EED-resolved final schedule")
        fig2, ax2 = plt.subplots(figsize=(6, 2 + 0.4 * len(all_machines_present)))
        gantt(ax2, "Final SBH schedule", lambda m: result["per_machine_schedule"][m])
        st.pyplot(fig2)
