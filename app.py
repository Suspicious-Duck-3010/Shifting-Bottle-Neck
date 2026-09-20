import io
import zipfile
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from sbh import Job, run_shifting_bottleneck, compute_heads_tails, draw_disjunctive_graph

PDF_PAGE_SIZE = (8.27, 11.69)  # A4 inches, portrait


def _pdf_title_page(pdf, jobs_list, result):
    fig = plt.figure(figsize=PDF_PAGE_SIZE)
    fig.text(0.5, 0.88, "Job Shop Scheduling Report", ha="center", fontsize=22, fontweight="bold")
    fig.text(0.5, 0.83, "Shifting Bottleneck Heuristic", ha="center", fontsize=15, color="#333333")
    fig.text(0.5, 0.78, datetime.now().strftime("Generated %Y-%m-%d %H:%M"),
             ha="center", fontsize=10, color="#777777")

    lines = [f"Jobs: {len(jobs_list)}",
             f"Machines: {len(sorted({m for j in jobs_list for m in j.routing}))}",
             "",
             "Routing summary:"]
    for j in jobs_list:
        routing_str = " -> ".join(f"M{m}" for m in j.routing)
        proc_str = ", ".join(str(p) for p in j.proc_times)
        lines.append(f"  Job {j.job_id}: {routing_str}   (Pij in routing order: {proc_str})")
    lines += ["", "Final results:",
              f"  Total makespan (Cmax): {result['final_cmax']}",
              f"  Maximum lateness: {result['max_lateness_overall']}"]
    fig.text(0.1, 0.68, "\n".join(lines), ha="left", va="top", fontsize=11, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _pdf_tables_page(pdf, title, sections, note=None):
    """sections: list of (subtitle, DataFrame)."""
    n = len(sections)
    fig, axes = plt.subplots(n, 1, figsize=PDF_PAGE_SIZE)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    for ax, (subtitle, df) in zip(axes, sections):
        ax.axis("off")
        if subtitle:
            ax.set_title(subtitle, fontsize=11, fontweight="bold", loc="left")
        tab = ax.table(cellText=df.astype(str).values, colLabels=list(df.columns),
                        cellLoc="center", loc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1, 1.5)
    if note:
        fig.text(0.5, 0.03, note, ha="center", fontsize=8, color="#666666", style="italic")
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def _pdf_graph_page(pdf, jobs_list, fixed_sequences, title):
    fig = draw_disjunctive_graph(jobs_list, fixed_sequences, title=title)
    fig.set_size_inches(*PDF_PAGE_SIZE[::-1])  # landscape orientation for wide graphs
    pdf.savefig(fig)
    plt.close(fig)


def build_report_pdf(jobs_list, all_machines_present, initial_state, result):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _pdf_title_page(pdf, jobs_list, result)

        job_rows = []
        for j in jobs_list:
            job_rows.append({
                "Job": j.job_id,
                "Routing": " -> ".join(f"M{m}" for m in j.routing),
                "Proc. times (routing order)": ", ".join(str(p) for p in j.proc_times),
                "Pij (increasing machine order)": ", ".join(
                    str(t) for _, t in sorted(zip(j.routing, j.proc_times))
                ),
            })
        _pdf_tables_page(pdf, "1. Job data", [(None, pd.DataFrame(job_rows))])

        machine_sections = []
        for m in all_machines_present:
            cols = [j.job_id for j in jobs_list if m in j.routing]
            data = {"": ["Pij", "rij", "dij"]}
            for jid in cols:
                data[f"Job {jid}"] = [
                    initial_state["durations"][(jid, m)],
                    initial_state["heads"][(jid, m)],
                    initial_state["deadlines"][(jid, m)],
                ]
            machine_sections.append((f"Machine {m}", pd.DataFrame(data)))
        _pdf_tables_page(
            pdf, "2. Per-machine Pij / rij / dij (initial graph)", machine_sections,
            note=f"Cmax before adding any disjunctive arcs: {initial_state['cmax']}",
        )

        _pdf_graph_page(pdf, jobs_list, {}, "Initial disjunctive graph -- no arcs fixed")

        for i, it in enumerate(result["iterations"], start=1):
            cand_sections = []
            for m, cand in it["candidates"].items():
                marker = "  <-- CHOSEN (bottleneck)" if m == it["chosen_machine"] else ""
                subtitle = f"Machine {m}: Lmax = {cand['lmax']}, sequence = {cand['seq']}{marker}"
                df = pd.DataFrame(cand["detail"]).rename(columns={
                    "job_id": "Job", "start": "Start", "finish": "Completion", "lateness": "Lateness",
                })
                cand_sections.append((subtitle, df))
            _pdf_tables_page(
                pdf, f"3. Shifting bottleneck -- Iteration {i}", cand_sections,
                note=f"Cmax going into this iteration: {it['cmax_before']}",
            )
            m = it["chosen_machine"]
            _pdf_graph_page(
                pdf, jobs_list, it["fixed_sequences_after"],
                f"Graph after fixing Machine {m} (sequence: {it['candidates'][m]['seq']})",
            )

        final_sections = []
        for m in result["all_machines"]:
            detail = result["per_machine_schedule"][m]
            seq = [d["job_id"] for d in detail]
            df = pd.DataFrame(detail).rename(columns={
                "job_id": "Job", "start": "Start", "finish": "Completion", "lateness": "Lateness",
            })
            final_sections.append((f"Machine {m} -- fixed sequence {seq}", df))
        _pdf_tables_page(
            pdf, "4. Final schedule per machine", final_sections,
            note=f"Total makespan (Cmax): {result['final_cmax']}   |   "
                 f"Maximum lateness: {result['max_lateness_overall']}",
        )

        _pdf_graph_page(
            pdf, jobs_list, result["fixed_sequences"],
            f"Final disjunctive graph -- all arcs resolved (Cmax = {result['final_cmax']})",
        )

    buf.seek(0)
    return buf.getvalue()


def fig_to_png_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def build_report_markdown(jobs_list, all_machines_present, initial_state, result):
    lines = []
    lines.append("# Job Shop Scheduling Report -- Shifting Bottleneck Heuristic")
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    lines.append("")

    lines.append("## 1. Job data")
    lines.append("")
    lines.append("| Job | Routing | Processing times (routing order) | Pij (increasing machine order) |")
    lines.append("|---|---|---|---|")
    for j in jobs_list:
        routing_str = " -> ".join(f"M{m}" for m in j.routing)
        proc_str = ", ".join(str(p) for p in j.proc_times)
        pij_str = ", ".join(str(t) for _, t in sorted(zip(j.routing, j.proc_times)))
        lines.append(f"| {j.job_id} | {routing_str} | {proc_str} | {pij_str} |")
    lines.append("")

    lines.append("## 2. Per-machine Pij / rij / dij (initial graph, before any disjunctive arcs)")
    lines.append("")
    for m in all_machines_present:
        cols = [j.job_id for j in jobs_list if m in j.routing]
        lines.append(f"**Machine {m}**")
        lines.append("")
        lines.append("| | " + " | ".join(f"Job {jid}" for jid in cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for label, key in (("Pij", "durations"), ("rij", "heads"), ("dij", "deadlines")):
            row = " | ".join(str(initial_state[key][(jid, m)]) for jid in cols)
            lines.append(f"| {label} | {row} |")
        lines.append("")
    lines.append(f"**Cmax before adding any disjunctive arcs: {initial_state['cmax']}**")
    lines.append("")
    lines.append("(See `graph_initial.png` for the disjunctive graph at this stage.)")
    lines.append("")

    lines.append("## 3. Shifting bottleneck iterations")
    lines.append("")
    for i, it in enumerate(result["iterations"], start=1):
        lines.append(f"### Iteration {i}")
        lines.append(f"Cmax going into this iteration: **{it['cmax_before']}**")
        lines.append("")
        for m, cand in it["candidates"].items():
            marker = "  <-- CHOSEN (bottleneck)" if m == it["chosen_machine"] else ""
            lines.append(f"- Machine {m}: best achievable Lmax = **{cand['lmax']}**, "
                          f"EDD-optimal sequence = {cand['seq']}{marker}")
            lines.append("")
            lines.append("  | Job | Start | Completion | Lateness |")
            lines.append("  |---|---|---|---|")
            for d in cand["detail"]:
                lines.append(f"  | {d['job_id']} | {d['start']} | {d['finish']} | {d['lateness']} |")
            lines.append("")
        lines.append(f"(See `graph_after_iteration_{i}_M{it['chosen_machine']}.png`.)")
        lines.append("")

    lines.append("## 4. Final schedule per machine")
    lines.append("")
    for m in result["all_machines"]:
        detail = result["per_machine_schedule"][m]
        seq = [d["job_id"] for d in detail]
        lines.append(f"**Machine {m}** -- fixed sequence: {seq}")
        lines.append("")
        lines.append("| Job | Start | Completion | Lateness |")
        lines.append("|---|---|---|---|")
        for d in detail:
            lines.append(f"| {d['job_id']} | {d['start']} | {d['finish']} | {d['lateness']} |")
        lines.append("")

    lines.append("## 5. Results")
    lines.append("")
    lines.append(f"- **Total makespan (Cmax): {result['final_cmax']}**")
    lines.append(f"- **Maximum lateness: {result['max_lateness_overall']}**")
    lines.append("")
    lines.append("(See `graph_final.png` for the fully-resolved disjunctive graph and "
                  "`gantt_final.png` for the schedule.)")
    lines.append("")

    return "\n".join(lines)


def build_report_zip(jobs_list, all_machines_present, initial_state, result):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        report_md = build_report_markdown(jobs_list, all_machines_present, initial_state, result)
        zf.writestr("report.md", report_md)

        fig_initial = draw_disjunctive_graph(jobs_list, {}, title="Initial graph -- no disjunctive arcs fixed")
        zf.writestr("graph_initial.png", fig_to_png_bytes(fig_initial))
        plt.close(fig_initial)

        for i, it in enumerate(result["iterations"], start=1):
            m = it["chosen_machine"]
            fig_it = draw_disjunctive_graph(
                jobs_list, it["fixed_sequences_after"],
                title=f"Graph after fixing Machine {m} "
                      f"(sequence: {it['candidates'][m]['seq']})",
            )
            zf.writestr(f"graph_after_iteration_{i}_M{m}.png", fig_to_png_bytes(fig_it))
            plt.close(fig_it)

        fig_final = draw_disjunctive_graph(
            jobs_list, result["fixed_sequences"],
            title=f"Final graph -- all disjunctive arcs resolved (Cmax = {result['final_cmax']})",
        )
        zf.writestr("graph_final.png", fig_to_png_bytes(fig_final))
        plt.close(fig_final)

    buf.seek(0)
    return buf.getvalue()

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
    if existing:
        # display processing times in increasing machine order (Pi1, Pi2, ...),
        # not in the job's chronological routing order
        default_proc = ",".join(
            str(t) for _, t in sorted(zip(existing.routing, existing.proc_times))
        )
    else:
        default_proc = ""

    routing_str = st.text_input(
        "Routing (machine numbers, chronological order, comma-separated)",
        value=default_routing, placeholder="e.g. 3,1,2",
    )
    proc_str = st.text_input(
        "Processing times (in INCREASING machine order, i.e. M1, M2, M3, ... "
        "comma-separated -- matches how Pij is normally given, regardless of routing order)",
        value=default_proc, placeholder="e.g. 7,6,9  (means M1=7, M2=6, M3=9)",
    )

    if st.button("Save job"):
        try:
            routing = [int(x.strip()) for x in routing_str.split(",") if x.strip() != ""]
            times_by_machine = [int(x.strip()) for x in proc_str.split(",") if x.strip() != ""]
            if len(routing) == 0 or len(routing) != len(times_by_machine):
                st.error("Routing and processing times must be the same, non-zero length.")
            elif len(set(routing)) != len(routing):
                st.error("A job cannot visit the same machine twice in this model.")
            else:
                # times_by_machine[k] is the processing time for the k-th smallest
                # machine number in this job's routing; re-map into routing order
                # for internal storage (Job.proc_times stays aligned with Job.routing).
                machine_time_map = dict(zip(sorted(routing), times_by_machine))
                proc_times = [machine_time_map[m] for m in routing]
                st.session_state.jobs[selected_id] = Job(selected_id, routing, proc_times)
                st.success(
                    f"Saved Job {selected_id}: routing {routing}, "
                    f"times in routing order {proc_times} "
                    f"(entered as M{{{','.join(f'{m}:{t}' for m, t in zip(sorted(routing), times_by_machine))}}})"
                )
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
            "Processing times (routing order)": ", ".join(str(p) for p in j.proc_times),
            "Pij (increasing machine order)": ", ".join(
                str(t) for _, t in sorted(zip(j.routing, j.proc_times))
            ),
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

    max_steps = max(len(j.routing) for j in jobs_list)

    def pad(seq):
        return list(seq) + [None] * (max_steps - len(seq))

    st.subheader("Routing matrix (chronological machine order per job)")
    st.dataframe(pd.DataFrame(
        {f"Job {j.job_id}": pad(j.routing) for j in jobs_list}
    ).T.rename(columns=lambda c: f"Step {c+1}"), use_container_width=True)

    st.subheader("Processing time matrix (Pij, in routing order)")
    st.dataframe(pd.DataFrame(
        {f"Job {j.job_id}": pad(j.proc_times) for j in jobs_list}
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

    st.subheader("Disjunctive graph (before scheduling any machine)")
    st.pyplot(draw_disjunctive_graph(jobs_list, {}, title="Initial graph -- no disjunctive arcs fixed"))
    st.caption("Solid arcs = conjunctive (job routing). Dashed, colored by machine = "
               "disjunctive pairs still unresolved.")

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

        st.markdown(f"**Graph after Machine {it['chosen_machine']} is scheduled:**")
        st.pyplot(draw_disjunctive_graph(
            jobs_list, it["fixed_sequences_after"],
            title=f"Graph after fixing Machine {it['chosen_machine']} "
                  f"(sequence: {it['candidates'][it['chosen_machine']]['seq']})",
        ))

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
    c1, c2, c3 = st.columns([1, 1, 1.4])
    c1.metric("Total makespan (Cmax)", result["final_cmax"])
    c2.metric("Maximum lateness", result["max_lateness_overall"])
    with c3:
        st.write("")
        report_pdf = build_report_pdf(jobs_list, all_machines_present, initial_state, result)
        st.download_button(
            "Export report (.pdf)",
            data=report_pdf,
            file_name="job_shop_sbh_report.pdf",
            mime="application/pdf",
            type="primary",
            help="Single PDF with all tables, matrices, the iteration log, the final schedule, "
                 "and every disjunctive graph (initial, after each iteration, final) as full pages.",
        )
        report_zip = build_report_zip(jobs_list, all_machines_present, initial_state, result)
        st.download_button(
            "Export report bundle (.zip: report.md + PNGs)",
            data=report_zip,
            file_name="job_shop_sbh_report.zip",
            mime="application/zip",
            help="Same content as the PDF, but as an editable markdown file plus separate PNG "
                 "images of each graph -- useful if you want to paste pieces into another document.",
        )

    st.subheader("Final disjunctive graph (all machines scheduled)")
    st.pyplot(draw_disjunctive_graph(
        jobs_list, result["fixed_sequences"],
        title=f"Final graph -- all disjunctive arcs resolved (Cmax = {result['final_cmax']})",
    ))

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
