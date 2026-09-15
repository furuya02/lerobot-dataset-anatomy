# lerobot-dataset-anatomy

LeRobot Dataset（`codebase_version: v2.1`）の中身を解剖して表示するスクリプトです。
**自分のデータセットに対して実行できます。** データセット本体はこのリポジトリに含まれていません。

分かること:

- `meta/info.json` が何を宣言していて、その実体がどのファイルにあるか
- `data/*.parquet` に**画像の列が無い**こと（画像は `videos/*.mp4` に別置き）
- `action`（リーダーが指令した角度）と `observation.state`（フォロワーが実際に到達した角度）の**違い**
- `videos/*.mp4` の実体（コーデック・解像度・フレーム数）が `info.json` の申告と合っているか
- `meta/episodes_stats.jsonl` が持つ正規化用の統計（画像の mean/std も含む）

## 動作確認した環境

| | |
|---|---|
| Python | 3.10.13 |
| pyarrow | 24.0.0 |
| PyAV | 14.4.0 |
| numpy | 2.2.6 |
| matplotlib | 3.10.5 |
| 対象データセット | lerobot 0.3.3 で収録した `codebase_version: v2.1` / `robot_type: so101_follower` |

> lerobot 本体はインストール不要です（parquet と mp4 を直接読みます）。
> `codebase_version` が `v2.1` でない場合は警告を出したうえで続行します。
> v3.0 はレイアウトが違う（複数エピソードを1ファイルに連結する）ため、そのままでは動きません。

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
`lerobot-record` で収録した場合は既定で
`~/.cache/huggingface/lerobot/<repo_id>` に置かれます。

```bash
D=~/.cache/huggingface/lerobot/local/duck_pickplace_real_20260814

python scripts/anatomy.py meta  $D              # meta/ を読む
python scripts/anatomy.py data  $D --episode 0  # parquet を読む
python scripts/anatomy.py video $D --episode 0  # mp4 の実体を読む
python scripts/anatomy.py diff  $D --episode 0  # action と observation.state を比べる
python scripts/anatomy.py all   $D              # 上を全部

# action と observation.state の時系列を図にする
python scripts/anatomy.py diff $D --episode 0 --plot action_vs_state.png
```

## 出力の例

50 エピソード / 9,995 フレームの実機データ（SO-ARM101 でアヒルを掴んでかごに入れる）での例です。

### `meta` — 画像だけ置き場所が違う

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

→ 9,995 フレーム分の関節角は**全部で 0.7 MB**。容量のほぼ全部が動画です。

### `data` — parquet に画像の列は無い

```
column                   arrow type                          info.json の dtype
action                   fixed_size_list<element: float>[6]  float32
observation.state        fixed_size_list<element: float>[6]  float32
timestamp                float                               float32
frame_index              int64                               int64
episode_index            int64                               int64
index                    int64                               int64
task_index               int64                               int64

画像らしき列: なし ← 画像は mp4 に別置き
```

### `video` — `info.json` の申告と mp4 の実体を突き合わせる

```
項目             mp4 の実体          info.json の申告
codec(tag)     av01             av1
decoder        libdav1d         (PyAV が選んだデコーダ)
pix_fmt        yuv420p          yuv420p
width          640              640
height         480              480
fps            15.0             15
frames         270              None
```

> PyAV の `codec_context.name` は「**選ばれたデコーダ名**」（AV1 なら `libdav1d`）を返し、
> ファイルに書かれているコーデックそのものではありません。実体は `codec_tag`（`av01`）で確認します。

### `diff` — `action` と `observation.state` は同じ [6] でも中身が違う

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
=== 追従の遅れ（action を k フレーム前にずらすと state に近づくか） ===
  0 フレーム遅らせる → 平均絶対差 2.373
  1 フレーム遅らせる → 平均絶対差 1.694
  2 フレーム遅らせる → 平均絶対差 1.082   ← 最小
  3 フレーム遅らせる → 平均絶対差 1.328
```

→ `action` は `observation.state` に対して**約2フレーム（≒133ms）先行**しています。
「指令」と「実測」であることが数値で確認できます。

## ライセンス

MIT
