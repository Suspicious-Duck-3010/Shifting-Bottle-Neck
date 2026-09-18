# Job Shop Scheduling - Shifting Bottleneck Heuristic

A Streamlit app for entering job-shop routing/processing-time data and
running the full (multi-iteration) shifting bottleneck heuristic on it.

Every number shown (heads `rij`, deadlines `dij`, one-machine EDD sequencing,
per-machine lateness, overall makespan) is computed deterministically in
[`sbh.py`](sbh.py) from whatever data you enter -- nothing is pre-filled.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. "New app" -> pick this repo/branch -> main file path `app.py` -> Deploy.

## How to use

1. Set the number of jobs.
2. For each job, pick it from the dropdown, type its routing (machine
   numbers, in the chronological order that job visits them) and its
   processing times in that same order. Save.
3. Check the drawn routing diagram to confirm it matches what you typed.
4. Once every job is saved, click **Simulate**.
5. Read off the routing/processing-time matrices, the per-machine
   `Pij/rij/dij` tables, each shifting-bottleneck iteration (which machine
   becomes the bottleneck and why), the final per-machine schedules, the
   overall makespan and maximum lateness, and the two Gantt-style flow
   diagrams (naive per-job routing vs. the final resolved schedule).

## Algorithm notes

- `rij` (release/head) and `dij` (deadline) are derived from longest-path
  (head/tail) calculations over the job shop's disjunctive graph, per the
  standard shifting-bottleneck formulation: `dij = Cmax - tail(i,j)`, where
  `tail(i,j)` is the sum of processing times of that job's operations that
  come *after* operation `(i,j)` in its own routing.
- Each machine's one-machine `1 || Lmax` subproblem is solved exactly by
  brute-force permutation (fine for course-sized instances; a few jobs per
  machine). The machine with the largest resulting `Lmax` is the bottleneck
  and gets its sequence fixed for that iteration; this repeats until every
  machine has a fixed sequence.
