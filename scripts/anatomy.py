#!/usr/bin/env python3
"""LeRobot Dataset (codebase_version v2.1) を解剖して中身を表示する。

使い方:
    python scripts/anatomy.py meta   <dataset_root>
    python scripts/anatomy.py data   <dataset_root> [--episode 0]
    python scripts/anatomy.py video  <dataset_root> [--episode 0]
    python scripts/anatomy.py diff   <dataset_root> [--episode 0] [--plot out.png]
    python scripts/anatomy.py all    <dataset_root>

dataset_root は meta/ data/ videos/ を直下に持つディレクトリ。
例: ~/.cache/huggingface/lerobot/local/duck_pickplace_real_20260814
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# --- 読み込みユーティリティ ------------------------------------------------

def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_info(root):
    return load_json(root / "meta" / "info.json")


def parquet_path(root, info, episode):
    """info.json の data_path をそのまま使って parquet の場所を決める。"""
    chunk = episode // info["chunks_size"]
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    return root / rel


def video_path(root, info, video_key, episode):
    chunk = episode // info["chunks_size"]
    rel = info["video_path"].format(
        episode_chunk=chunk, video_key=video_key, episode_index=episode
    )
    return root / rel


def video_keys(info):
    return [k for k, v in info["features"].items() if v["dtype"] == "video"]


def head(title):
    print(f"\n=== {title} " + "=" * max(0, 68 - len(title)))


# --- meta ------------------------------------------------------------------

def cmd_meta(root, args):
    info = load_info(root)

    head("meta/info.json")
    for k in ("codebase_version", "robot_type", "fps", "total_episodes",
              "total_frames", "total_tasks", "total_videos", "total_chunks",
              "chunks_size"):
        print(f"{k:20s} {info.get(k)}")
    print(f"{'data_path':20s} {info.get('data_path')}")
    print(f"{'video_path':20s} {info.get('video_path')}")
    print(f"{'splits':20s} {info.get('splits')}")

    head("features（info.json が宣言している列と、その実体の置き場所）")
    print(f"{'name':32s} {'dtype':8s} {'shape':16s} where")
    for name, f in info["features"].items():
        where = "videos/*.mp4" if f["dtype"] == "video" else "data/*.parquet"
        print(f"{name:32s} {f['dtype']:8s} {str(f['shape']):16s} {where}")

    head("画像特徴量の video 情報（info.json 側の申告値）")
    for k in video_keys(info):
        vi = info["features"][k].get("info", {})
        print(f"{k}")
        for kk in sorted(vi):
            print(f"    {kk:24s} {vi[kk]}")

    head("meta/tasks.jsonl（言語指示の実体）")
    for t in load_jsonl(root / "meta" / "tasks.jsonl"):
        print(f"  task_index={t['task_index']}  {t['task']!r}")

    head("meta/episodes.jsonl（エピソード一覧）")
    eps = load_jsonl(root / "meta" / "episodes.jsonl")
    lengths = [e["length"] for e in eps]
    fps = info["fps"]
    print(f"エピソード数       {len(eps)}")
    print(f"総フレーム数       {sum(lengths)}  (info.json: {info['total_frames']})")
    med, avg = statistics.median(lengths), statistics.mean(lengths)
    print(f"長さ min/median/mean/max {min(lengths)} / {med:g} / {avg:.1f} / {max(lengths)} フレーム")
    print(f"                         {min(lengths)/fps:.1f} / {med/fps:.1f} / {avg/fps:.1f} / {max(lengths)/fps:.1f} 秒 (fps={fps})")
    print(f"総時間             {sum(lengths)/fps/60:.1f} 分")
    print("先頭3件:")
    for e in eps[:3]:
        print(f"  {e}")

    head("meta/episodes_stats.jsonl（正規化用の統計）")
    stats = load_jsonl(root / "meta" / "episodes_stats.jsonl")
    print(f"行数 {len(stats)}（エピソードごとに1行）")
    st0 = stats[0]["stats"]
    print(f"統計を持っている特徴量: {len(st0)} 個")
    for name in st0:
        print(f"  - {name}  keys={sorted(st0[name])}")
    print("\nepisode 0 の action の統計:")
    for k, v in st0["action"].items():
        v = np.array(v).ravel()
        print(f"  {k:6s} {np.round(v, 3).tolist()}")
    img = next((k for k in st0 if k.startswith("observation.images")), None)
    if img:
        print(f"\nepisode 0 の {img} の統計（画像も統計を持っている）:")
        for k in ("mean", "std"):
            v = np.array(st0[img][k]).ravel()
            print(f"  {k:6s} {np.round(v, 4).tolist()}  ← チャンネルごと(RGB)")

    head("ファイルの実数")
    for d in ("meta", "data", "videos"):
        files = sorted(p for p in (root / d).rglob("*") if p.is_file())
        size = sum(p.stat().st_size for p in files)
        print(f"{d:8s} {len(files):4d} files  {size/1024/1024:8.1f} MB")
    all_files = [p for p in root.rglob("*") if p.is_file()]
    all_dirs = [p for p in root.rglob("*") if p.is_dir()]
    print(f"{'合計':8s} {len(all_files):4d} files + {len(all_dirs)} dirs  "
          f"{sum(p.stat().st_size for p in all_files)/1024/1024:.1f} MB")


# --- data ------------------------------------------------------------------

def cmd_data(root, args):
    info = load_info(root)
    ep = args.episode
    path = parquet_path(root, info, ep)

    head(f"data: episode {ep} の parquet")
    print(f"path  {path.relative_to(root)}")
    print(f"size  {path.stat().st_size/1024:.1f} KB")

    table = pq.read_table(path)
    print(f"rows  {table.num_rows}   cols {table.num_columns}")

    head("列の構成（★画像の列が無いことを確認する）")
    print(f"{'column':24s} {'arrow type':24s} info.json の dtype")
    for f in table.schema:
        declared = info["features"].get(f.name, {}).get("dtype", "-")
        print(f"{f.name:24s} {str(f.type):24s} {declared}")
    img_cols = [f.name for f in table.schema if "image" in f.name]
    print(f"\n画像らしき列: {img_cols if img_cols else 'なし ← 画像は mp4 に別置き'}")

    head("先頭3フレーム")
    df = table.to_pandas()
    for i in range(min(3, len(df))):
        r = df.iloc[i]
        print(f"[frame {i}]")
        for c in df.columns:
            v = r[c]
            v = np.round(np.asarray(v, dtype=float), 3).tolist() if hasattr(v, "__len__") else v
            print(f"    {c:20s} {v}")

    head("timestamp の刻み（fps との整合）")
    ts = np.asarray(df["timestamp"], dtype=float)
    d = np.diff(ts)
    print(f"fps={info['fps']} → 期待される間隔 {1/info['fps']:.6f} s")
    print(f"実際の間隔 min={d.min():.6f} max={d.max():.6f} 一意な値={np.unique(np.round(d, 9)).tolist()[:5]}")

    head("全 parquet の合計行数（info.json の total_frames と突き合わせる）")
    total = 0
    for i in range(info["total_episodes"]):
        total += pq.read_metadata(parquet_path(root, info, i)).num_rows
    print(f"合計 {total} 行  /  info.json total_frames = {info['total_frames']}  "
          f"→ {'一致' if total == info['total_frames'] else '不一致'}")


# --- video -----------------------------------------------------------------

def cmd_video(root, args):
    import av

    info = load_info(root)
    ep = args.episode

    for key in video_keys(info):
        path = video_path(root, info, key, ep)
        head(f"video: {key} / episode {ep}")
        print(f"path  {path.relative_to(root)}")
        print(f"size  {path.stat().st_size/1024:.1f} KB")

        with av.open(str(path)) as c:
            v = c.streams.video[0]
            declared = info["features"][key].get("info", {})
            # PyAV の codec_context.name は「選ばれたデコーダ名」（AV1 なら libdav1d）で、
            # ファイルに書かれているコーデックそのものではない。実体は codec_tag（av01）を見る。
            rows = [
                ("codec(tag)", v.codec_context.codec_tag, declared.get("video.codec")),
                ("decoder", v.codec_context.name, "(PyAV が選んだデコーダ)"),
                ("pix_fmt", v.codec_context.pix_fmt, declared.get("video.pix_fmt")),
                ("width", v.codec_context.width, declared.get("video.width")),
                ("height", v.codec_context.height, declared.get("video.height")),
                ("fps", float(v.average_rate), declared.get("video.fps")),
                ("frames", v.frames, None),
                ("duration(s)", round(float(v.duration * v.time_base), 3) if v.duration else None, None),
                ("audio streams", len(c.streams.audio), declared.get("video.has_audio")),
            ]
            print(f"\n{'項目':14s} {'mp4 の実体':16s} info.json の申告")
            for name, actual, dec in rows:
                print(f"{name:14s} {str(actual):16s} {dec}")

        n_rows = pq.read_metadata(parquet_path(root, info, ep)).num_rows
        print(f"\nparquet の行数 {n_rows} と mp4 のフレーム数を突き合わせる "
              f"→ 同じなら 1 行 = 1 フレームで対応がとれている")


# --- diff（action と observation.state の違い） -----------------------------

def cmd_diff(root, args):
    info = load_info(root)
    ep = args.episode
    df = pq.read_table(parquet_path(root, info, ep)).to_pandas()

    a = np.stack(df["action"].to_numpy())            # [T, 6] リーダーが指令した角度
    s = np.stack(df["observation.state"].to_numpy())  # [T, 6] フォロワーが実際に到達した角度
    names = info["features"]["action"]["names"]
    if isinstance(names, dict):
        names = names.get("motors", list(names))

    head(f"action vs observation.state（episode {ep} / {len(df)} フレーム）")
    print(f"shape  action={a.shape}  observation.state={s.shape}  ← 次元も名前も同じ")

    head("第1フレーム（同じ [6] でも値が違う）")
    print(f"{'joint':18s} {'action':>10s} {'obs.state':>10s} {'diff':>9s}")
    for i, n in enumerate(names):
        print(f"{n:18s} {a[0, i]:10.3f} {s[0, i]:10.3f} {a[0, i]-s[0, i]:9.3f}")

    head("エピソード全体の差（action - observation.state）")
    d = a - s
    ts = np.asarray(df["timestamp"], dtype=float)
    print(f"{'joint':18s} {'mean':>9s} {'std':>9s} {'max|diff|':>10s}  {'その時刻':>9s}")
    for i, n in enumerate(names):
        j = int(np.abs(d[:, i]).argmax())
        print(f"{n:18s} {d[:, i].mean():9.3f} {d[:, i].std():9.3f} "
              f"{np.abs(d[:, i]).max():10.3f}  t={ts[j]:6.2f}s")
    print("\n差が大きい時刻 = 指令どおりに動けていない瞬間。"
          "\ngripper なら「物を掴んで閉じきれない」、他の関節なら「可動域や負荷で追いつけない」。")

    head("追従の遅れ（action を k フレーム前にずらすと state に近づくか）")
    for k in range(0, 5):
        err = np.abs(a[:len(a)-k] - s[k:]).mean()
        print(f"  {k} フレーム遅らせる → 平均絶対差 {err:.3f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.asarray(df["timestamp"], dtype=float)
        fig, axes = plt.subplots(len(names), 1, figsize=(9, 1.7 * len(names)), sharex=True)
        for i, n in enumerate(names):
            axes[i].plot(t, a[:, i], label="action", lw=1.2)
            axes[i].plot(t, s[:, i], label="observation.state", lw=1.2, ls="--")
            axes[i].set_ylabel(n, fontsize=8)
            axes[i].tick_params(labelsize=8)
        axes[0].legend(fontsize=8, ncol=2)
        axes[-1].set_xlabel("timestamp [s]")
        fig.suptitle(f"action vs observation.state (episode {ep})")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"\n図を書き出しました: {args.plot}")


# --- entry -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("meta", "data", "video", "diff", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("root", type=Path)
        if name != "meta":
            sp.add_argument("--episode", type=int, default=0)
        if name in ("diff", "all"):
            sp.add_argument("--plot", default=None, help="PNG の出力先")
    args = p.parse_args()

    root = args.root.expanduser()
    info = load_info(root)
    if info["codebase_version"] != "v2.1":
        print(f"警告: このスクリプトは v2.1 用です（このデータセットは "
              f"{info['codebase_version']}）\n")

    cmds = {"meta": cmd_meta, "data": cmd_data, "video": cmd_video, "diff": cmd_diff}
    if args.cmd == "all":
        for name in ("meta", "data", "video", "diff"):
            cmds[name](root, args)
    else:
        cmds[args.cmd](root, args)


if __name__ == "__main__":
    main()
