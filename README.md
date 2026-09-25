# lerobot-dataset-anatomy

Scripts that dissect a LeRobot Dataset and print what is actually inside.
**Supports both `v3.0` (current) and `v2.1` (legacy).**

**Point them at your own dataset.** No dataset is bundled with this repository, but the two
datasets used for the examples below are published on the Hugging Face Hub (CC BY 4.0):

- v3.0: [furuya02/so101-duck-pickup-v30](https://huggingface.co/datasets/furuya02/so101-duck-pickup-v30) — 30 episodes, 1 camera, H.264
- v2.1: [furuya02/so101-duck-pickplace-v21](https://huggingface.co/datasets/furuya02/so101-duck-pickplace-v21) — 50 episodes, 2 cameras, AV1

```bash
hf download furuya02/so101-duck-pickup-v30 --repo-type dataset --local-dir ./duck_v30
python scripts/anatomy.py all ./duck_v30 --episode 5
```

What you get:

- What `meta/info.json` declares, and which file actually holds each declared feature
- That `data/*.parquet` has **no image column** (images live in `videos/*.mp4`)
- **Where each episode begins and ends** — in v3.0 many episodes are concatenated into one
  file, and the boundaries live in `meta/episodes/`, not in the file names
- The difference between `action` (angle commanded by the leader arm) and `observation.state`
  (angle the follower arm actually reached), including the follow-up lag
- Whether the real `videos/*.mp4` (codec, resolution, frame count) matches what `info.json` claims

## v3.0 and v2.1

`lerobot v0.4.0` switched the dataset format from `v2.1` to `v3.0`.
The scripts detect `codebase_version` and adapt.

| | v2.1 | v3.0 |
|---|---|---|
| Episode ↔ file | 1 episode = 1 parquet + 1 mp4 per camera | many episodes concatenated per file |
| Episode boundaries | encoded in the file name | `meta/episodes/**.parquet` (`dataset_from_index` / `from_timestamp`) |
| Tasks | `meta/tasks.jsonl` | `meta/tasks.parquet` |
| Episode metadata | `episodes.jsonl` + `episodes_stats.jsonl` | `meta/episodes/**.parquet` (stats included) |
| Dataset-wide stats | none (aggregated on load) | `meta/stats.json` (with `q01`…`q99`) |
| Path template keys | `{episode_chunk}` / `{episode_index}` | `{chunk_index}` / `{file_index}` |

Note that `lerobot >= 0.4.0` refuses to load a `v2.1` dataset
(`BackwardCompatibilityError`) and asks you to convert it. These scripts read the files
directly, so they work on both without converting anything.

## Verified environment

| | |
|---|---|
| Python | 3.10.13 |
| pyarrow | 24.0.0 |
| PyAV | 14.4.0 |
| numpy | 2.2.6 |
| matplotlib | 3.10.5 |
| Tested against | a v3.0 dataset (30 episodes / 6,977 frames, 1 camera, h264) and a v2.1 dataset (50 episodes / 9,995 frames, 2 cameras, AV1) |

> lerobot itself is not required — the scripts read the parquet and mp4 files directly.

## Setup

```bash
git clone https://github.com/furuya02/lerobot-dataset-anatomy.git
cd lerobot-dataset-anatomy

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

`<dataset_root>` is the directory that directly contains `meta/`, `data/` and `videos/`.
Datasets recorded with `lerobot-record` land in `~/.cache/huggingface/lerobot/<repo_id>` by default.

```bash
D=~/.cache/huggingface/lerobot/<repo_id>

python scripts/anatomy.py meta  $D              # read meta/
python scripts/anatomy.py data  $D --episode 5  # read the parquet
python scripts/anatomy.py video $D --episode 5  # read the real mp4
python scripts/anatomy.py diff  $D --episode 5  # compare action and observation.state
python scripts/anatomy.py all   $D              # all of the above

# plot action vs observation.state over time
python scripts/anatomy.py diff $D --episode 5 --plot action_vs_state.png
```

## Example output (v3.0)

Taken from a real dataset of 30 episodes / 6,977 frames
(SO-ARM101 picking up a rubber duck and placing it into a basket).

### `meta` — only the images are stored somewhere else

```
name                             dtype    shape            where
action                           float32  [6]              data/*.parquet
observation.state                float32  [6]              data/*.parquet
observation.images.front         video    [480, 640, 3]    videos/*.mp4
timestamp                        float32  [1]              data/*.parquet
...
```

```
meta        4 files       0.1 MB
data        1 files       0.4 MB
videos      2 files     243.3 MB
合計          7 files     243.9 MB
```

→ All joint angles for 6,977 frames take **0.4 MB in total**. Almost the entire size is video.

### `meta` — episodes are located through metadata, not file names

```
  ep  length    data file  from_index  to_index   video file   from_ts     to_ts
   0     235     file-000           0       235     file-000     0.000    15.667
   1     216     file-000         235       451     file-000    15.667    30.067
  28     249     file-000        6456      6705     file-001    74.733    91.333
  29     272     file-000        6705      6977     file-001    91.333   109.467
```

A single `file-000.parquet` holds all 30 episodes; `file-000.mp4` fills up at ~200 MB and
the remaining episodes continue in `file-001.mp4` with timestamps restarting from zero.

### `data` — there is no image column in the parquet

```
column                   arrow type                           dtype in info.json
action                   fixed_size_list<element: float>[6]   float32
observation.state        fixed_size_list<element: float>[6]   float32
timestamp                float                                float32
frame_index              int64                                int64
...

image-like columns: none — images are kept separately as mp4
```

### `video` — cross-check the real mp4 against what `info.json` declares

```
field            real mp4         declared in info.json
codec(tag)     avc1             h264
decoder        h264             (decoder picked by PyAV)
pix_fmt        yuv420p          yuv420p
width          640              640
frames         1642             None
```

```
episode 29 spans 91.333 s … 109.467 s = 272 frames
parquet 272 rows / 272 frames in that span → match
```

> PyAV's `codec_context.name` returns the **name of the decoder that was selected**
> (`libdav1d` for AV1), not the codec written in the file. The authoritative value is
> `codec_tag` (`av01` for AV1, `avc1` for H.264).

### `diff` — `action` and `observation.state` are both [6], but they are not the same thing

```
joint                  action  obs.state      diff
shoulder_pan.pos       -7.077     -2.374    -4.703
shoulder_lift.pos    -105.407   -105.231    -0.176
...
gripper.pos             0.159      1.824    -1.665
```

```
=== lag (shift action k frames and see if it matches state better) ===
  shift by 0 frames -> mean abs diff 1.893
  shift by 1 frames -> mean abs diff 1.312
  shift by 2 frames -> mean abs diff 0.791   <- minimum
  shift by 3 frames -> mean abs diff 1.000
```

→ `action` leads `observation.state` by roughly **2 frames (~133 ms)**.
Across 80 episodes from two different datasets (v3.0 and v2.1, different machines),
the minimum was at k = 2 every single time.

## License

MIT
