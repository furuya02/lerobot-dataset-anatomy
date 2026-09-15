# lerobot-dataset-anatomy

Scripts that dissect a LeRobot Dataset (`codebase_version: v2.1`) and print what is actually inside.
**Point them at your own dataset.** No dataset is bundled with this repository.

What you get:

- What `meta/info.json` declares, and which file actually holds each declared feature
- That `data/*.parquet` has **no image column** (images live in `videos/*.mp4`)
- The difference between `action` (angle commanded by the leader arm) and `observation.state`
  (angle the follower arm actually reached)
- Whether the real `videos/*.mp4` (codec, resolution, frame count) matches what `info.json` claims
- The normalization statistics in `meta/episodes_stats.jsonl` (including per-channel image mean/std)

## Verified environment

| | |
|---|---|
| Python | 3.10.13 |
| pyarrow | 24.0.0 |
| PyAV | 14.4.0 |
| numpy | 2.2.6 |
| matplotlib | 3.10.5 |
| Target dataset | recorded with lerobot 0.3.3, `codebase_version: v2.1`, `robot_type: so101_follower` |

> lerobot itself is not required — the scripts read the parquet and mp4 files directly.
> If `codebase_version` is not `v2.1` a warning is printed and the run continues.
> v3.0 uses a different layout (several episodes concatenated into one file) and will not work as is.

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
D=~/.cache/huggingface/lerobot/local/duck_pickplace_real_20260814

python scripts/anatomy.py meta  $D              # read meta/
python scripts/anatomy.py data  $D --episode 0  # read the parquet
python scripts/anatomy.py video $D --episode 0  # read the real mp4
python scripts/anatomy.py diff  $D --episode 0  # compare action and observation.state
python scripts/anatomy.py all   $D              # all of the above

# plot action vs observation.state over time
python scripts/anatomy.py diff $D --episode 0 --plot action_vs_state.png
```

## Example output

Taken from a real dataset of 50 episodes / 9,995 frames
(SO-ARM101 picking up a rubber duck and placing it into a basket).

### `meta` — only the images are stored somewhere else

```
name                             dtype    shape            where
action                           float32  [6]              data/*.parquet
observation.state                float32  [6]              data/*.parquet
observation.images.front         video    [480, 640, 3]    videos/*.mp4
observation.images.wrist         video    [480, 640, 3]    videos/*.mp4
timestamp                        float32  [1]              data/*.parquet
...
```

```
meta        4 files       0.1 MB
data       50 files       0.7 MB
videos    100 files     182.1 MB
```

→ All joint angles for 9,995 frames take **0.7 MB in total**. Almost the entire size is video.

### `data` — there is no image column in the parquet

```
column                   arrow type                          dtype in info.json
action                   fixed_size_list<element: float>[6]  float32
observation.state        fixed_size_list<element: float>[6]  float32
timestamp                float                               float32
frame_index              int64                               int64
episode_index            int64                               int64
index                    int64                               int64
task_index               int64                               int64

image-like columns: none — images are kept separately as mp4
```

### `video` — cross-check the real mp4 against what `info.json` declares

```
field            real mp4         declared in info.json
codec(tag)     av01             av1
decoder        libdav1d         (decoder picked by PyAV)
pix_fmt        yuv420p          yuv420p
width          640              640
height         480              480
fps            15.0             15
frames         270              None
```

> PyAV's `codec_context.name` returns the **name of the decoder that was selected**
> (`libdav1d` for AV1), not the codec written in the file. The authoritative value is
> `codec_tag` (`av01`).

### `diff` — `action` and `observation.state` are both [6], but they are not the same thing

```
joint                  action  obs.state      diff
shoulder_pan.pos        3.736      3.385     0.352
shoulder_lift.pos    -105.231   -104.791    -0.440
elbow_flex.pos         96.747     96.703     0.044
wrist_flex.pos         88.527     85.099     3.429
wrist_roll.pos         12.308     12.088     0.220
gripper.pos             0.308      0.898    -0.590
```

```
=== lag (shift action k frames and see if it matches state better) ===
  shift by 0 frames -> mean abs diff 2.373
  shift by 1 frames -> mean abs diff 1.694
  shift by 2 frames -> mean abs diff 1.082   <- minimum
  shift by 3 frames -> mean abs diff 1.328
```

→ `action` leads `observation.state` by roughly **2 frames (~133 ms)**.
The "command vs. measurement" relationship is visible in the numbers.

## License

MIT
