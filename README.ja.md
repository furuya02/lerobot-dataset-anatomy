# lerobot-dataset-anatomy

LeRobot Dataset を解剖して、中身に何が入っているかを表示するスクリプトです。
**`v3.0`（現行）と `v2.1`（旧形式）の両方に対応**しています。

**自分のデータセットに当てて使ってください。** このリポジトリにデータセット本体は含まれていません。

分かること:

- `meta/info.json` が宣言している特徴量と、その実体がどのファイルにあるか
- `data/*.parquet` に**画像の列が無い**こと（画像は `videos/*.mp4` にある）
- **各エピソードがどこから始まりどこで終わるか** — v3.0 では複数エピソードが 1 ファイルに
  連結されており、その境界はファイル名ではなく `meta/episodes/` にあります
- `action`（リーダーアームが指令した角度）と `observation.state`（フォロワーが実際に到達した角度）
  の違い、および追従の遅れ
- 実際の `videos/*.mp4`（コーデック・解像度・フレーム数）が `info.json` の申告と一致するか

## v3.0 と v2.1

`lerobot v0.4.0` でデータセット形式が `v2.1` から `v3.0` に変わりました。
スクリプトは `codebase_version` を見て自動的に切り替えます。

| | v2.1 | v3.0 |
|---|---|---|
| エピソードとファイル | 1 エピソード = 1 parquet + カメラごとに 1 mp4 | 複数エピソードを 1 ファイルに連結 |
| エピソードの境界 | ファイル名に埋め込まれている | `meta/episodes/**.parquet`（`dataset_from_index` / `from_timestamp`） |
| タスク | `meta/tasks.jsonl` | `meta/tasks.parquet` |
| エピソードのメタ | `episodes.jsonl` + `episodes_stats.jsonl` | `meta/episodes/**.parquet`（統計も同居） |
| 全体統計 | 無し（読み込み時に集約） | `meta/stats.json`（`q01`〜`q99` の分位数付き） |
| パステンプレートの変数 | `{episode_chunk}` / `{episode_index}` | `{chunk_index}` / `{file_index}` |

なお `lerobot >= 0.4.0` は `v2.1` のデータセットを読み込もうとすると
`BackwardCompatibilityError` で停止し、変換を促します。本スクリプトはファイルを直接読むため、
変換しなくても両方そのまま解析できます。

## 動作確認環境

| | |
|---|---|
| Python | 3.10.13 |
| pyarrow | 24.0.0 |
| PyAV | 14.4.0 |
| numpy | 2.2.6 |
| matplotlib | 3.10.5 |
| 検証したデータセット | v3.0（30 エピソード / 6,977 フレーム、カメラ 1 台、h264）と v2.1（50 エピソード / 9,995 フレーム、カメラ 2 台、AV1） |

> lerobot 本体は不要です。parquet と mp4 を直接読んでいます。

## セットアップ

```bash
git clone https://github.com/furuya02/lerobot-dataset-anatomy.git
cd lerobot-dataset-anatomy

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使い方

`<dataset_root>` は `meta/` `data/` `videos/` を直下に持つディレクトリです。
`lerobot-record` で収録したデータセットは、既定では `~/.cache/huggingface/lerobot/<repo_id>` に置かれます。

```bash
D=~/.cache/huggingface/lerobot/<repo_id>

python scripts/anatomy.py meta  $D              # meta/ を読む
python scripts/anatomy.py data  $D --episode 5  # parquet を読む
python scripts/anatomy.py video $D --episode 5  # mp4 の実体を読む
python scripts/anatomy.py diff  $D --episode 5  # action と observation.state を比較
python scripts/anatomy.py all   $D              # すべて

# action と observation.state の時系列を図にする
python scripts/anatomy.py diff $D --episode 5 --plot action_vs_state.png
```

## 出力例（v3.0）

30 エピソード / 6,977 フレームの実データ（SO-ARM101 でアヒルを掴んでかごに入れるタスク）での出力です。

### `meta` — 画像だけが別の場所にある

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

→ 6,977 フレーム分の関節角は**全部で 0.4 MB**。容量のほとんどは動画です。

### `meta` — エピソードはファイル名ではなくメタデータで特定する

```
  ep  length    data file  from_index  to_index   video file   from_ts     to_ts
   0     235     file-000           0       235     file-000     0.000    15.667
   1     216     file-000         235       451     file-000    15.667    30.067
  28     249     file-000        6456      6705     file-001    74.733    91.333
  29     272     file-000        6705      6977     file-001    91.333   109.467
```

`file-000.parquet` 1 つに 30 エピソードすべてが入っています。`file-000.mp4` は約 200 MB で
いっぱいになり、残りのエピソードは `file-001.mp4` に続きます（タイムスタンプは 0 から振り直し）。

### `data` — parquet に画像の列は無い

```
column                   arrow type                           dtype in info.json
action                   fixed_size_list<element: float>[6]   float32
observation.state        fixed_size_list<element: float>[6]   float32
timestamp                float                                float32
frame_index              int64                                int64
...

画像らしき列: なし ← 画像は mp4 に別置き
```

### `video` — mp4 の実体と `info.json` の申告を突き合わせる

```
項目             mp4 の実体          info.json の申告
codec(tag)     avc1             h264
decoder        h264             (PyAV が選んだデコーダ)
pix_fmt        yuv420p          yuv420p
width          640              640
frames         1642             None
```

```
episode 29 の区間  91.333 s 〜 109.467 s  = 272 フレーム相当
parquet 272 行 / 区間 272 フレーム → 一致
```

> PyAV の `codec_context.name` は**選ばれたデコーダの名前**（AV1 なら `libdav1d`）を返すもので、
> ファイルに書かれているコーデックそのものではありません。実体は `codec_tag`
> （AV1 なら `av01`、H.264 なら `avc1`）で見ます。

### `diff` — `action` と `observation.state` は同じ [6] でも別のもの

```
joint                  action  obs.state      diff
shoulder_pan.pos       -7.077     -2.374    -4.703
shoulder_lift.pos    -105.407   -105.231    -0.176
...
gripper.pos             0.159      1.824    -1.665
```

```
=== 追従の遅れ（action を k フレームずらすと state に近づくか） ===
  0 フレーム遅らせる → 平均絶対差 1.893
  1 フレーム遅らせる → 平均絶対差 1.312
  2 フレーム遅らせる → 平均絶対差 0.791   ← 最小
  3 フレーム遅らせる → 平均絶対差 1.000
```

→ `action` は `observation.state` に約 **2 フレーム（≒133 ms）先行**しています。
異なるマシンで収録した 2 つのデータセット（v3.0 と v2.1）の 80 エピソードすべてで、
最小になったのは k = 2 でした。

## ライセンス

MIT
