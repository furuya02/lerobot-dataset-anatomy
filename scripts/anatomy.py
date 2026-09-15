#!/usr/bin/env python3
"""LeRobot Dataset を解剖して中身を表示する（v3.0 / v2.1 両対応）。

使い方:
    python scripts/anatomy.py meta   <dataset_root>
    python scripts/anatomy.py data   <dataset_root> [--episode 0]
    python scripts/anatomy.py video  <dataset_root> [--episode 0]
    python scripts/anatomy.py diff   <dataset_root> [--episode 0] [--plot out.png]
    python scripts/anatomy.py all    <dataset_root>

dataset_root は meta/ data/ videos/ を直下に持つディレクトリ。
例: ~/.cache/huggingface/lerobot/<repo_id>

v3.0 と v2.1 の最大の違いは「1 エピソードが 1 ファイルかどうか」。
v2.1 はファイル名でエピソードを特定できるが、v3.0 は複数エピソードを 1 ファイルに連結し、
どこからどこまでが何番のエピソードかを meta/episodes/ に持つ。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def head(title):
    print(f"\n=== {title} " + "=" * max(0, 68 - len(title)))


class DatasetFiles:
    """meta/ data/ videos/ を直接読む最小の読み取り層。

    lerobot の API は使わない。v2.1 と v3.0 のレイアウトの違いはここで吸収する。
    """

    def __init__(self, root):
        self.root = Path(root).expanduser()
        self.info = load_json(self.root / "meta" / "info.json")
        self.version = self.info["codebase_version"]
        if self.version not in ("v3.0", "v2.1"):
            raise SystemExit(f"未対応の codebase_version: {self.version}")

    # --- 共通 ---------------------------------------------------------------

    @property
    def fps(self):
        return self.info["fps"]

    def video_keys(self):
        return [k for k, v in self.info["features"].items() if v["dtype"] == "video"]

    def files(self):
        """(ディレクトリ, ファイル数, バイト数) の一覧。"""
        out = []
        for d in ("meta", "data", "videos"):
            fs = [p for p in (self.root / d).rglob("*") if p.is_file()]
            out.append((d, len(fs), sum(p.stat().st_size for p in fs)))
        return out

    # --- エピソード一覧 -----------------------------------------------------

    def episodes(self):
        """[{episode_index, length, tasks, ...}] を episode_index 順で返す。"""
        if self.version == "v2.1":
            return load_jsonl(self.root / "meta" / "episodes.jsonl")
        rows = []
        for p in sorted((self.root / "meta" / "episodes").rglob("*.parquet")):
            rows += pq.read_table(p).to_pandas().to_dict("records")
        return sorted(rows, key=lambda r: r["episode_index"])

    def episode(self, ep):
        for r in self.episodes():
            if r["episode_index"] == ep:
                return r
        raise SystemExit(f"episode {ep} が meta に見つからない")

    # --- 言語指示 -----------------------------------------------------------

    def tasks(self):
        """{task_index: task 文字列}"""
        if self.version == "v2.1":
            return {r["task_index"]: r["task"] for r in load_jsonl(self.root / "meta" / "tasks.jsonl")}
        df = pq.read_table(self.root / "meta" / "tasks.parquet").to_pandas().reset_index()
        return {int(r["task_index"]): r["task"] for _, r in df.iterrows()}

    # --- parquet ------------------------------------------------------------

    def data_files(self):
        if self.version == "v2.1":
            return [self._v21_parquet(e["episode_index"]) for e in self.episodes()]
        seen, out = set(), []
        for e in self.episodes():
            key = (e["data/chunk_index"], e["data/file_index"])
            if key not in seen:
                seen.add(key)
                out.append(self.root / self.info["data_path"].format(
                    chunk_index=key[0], file_index=key[1]))
        return out

    def _v21_parquet(self, ep):
        chunk = ep // self.info["chunks_size"]
        return self.root / self.info["data_path"].format(episode_chunk=chunk, episode_index=ep)

    def parquet_of(self, ep):
        """エピソード ep の行が入っている parquet のパス。"""
        if self.version == "v2.1":
            return self._v21_parquet(ep)
        e = self.episode(ep)
        return self.root / self.info["data_path"].format(
            chunk_index=e["data/chunk_index"], file_index=e["data/file_index"])

    def frames(self, ep):
        """エピソード ep の行だけを DataFrame で返す。

        v2.1 はファイルがエピソード単位なので全行。
        v3.0 は 1 ファイルに複数エピソードが入っているので episode_index で絞る。
        """
        df = pq.read_table(self.parquet_of(ep)).to_pandas()
        if self.version == "v3.0":
            df = df[df["episode_index"] == ep].reset_index(drop=True)
        return df

    # --- 動画 ---------------------------------------------------------------

    def video_of(self, key, ep):
        """(パス, from_timestamp, to_timestamp) を返す。v2.1 は区間が無いので None。"""
        if self.version == "v2.1":
            chunk = ep // self.info["chunks_size"]
            p = self.root / self.info["video_path"].format(
                episode_chunk=chunk, video_key=key, episode_index=ep)
            return p, None, None
        e = self.episode(ep)
        p = self.root / self.info["video_path"].format(
            video_key=key,
            chunk_index=e[f"videos/{key}/chunk_index"],
            file_index=e[f"videos/{key}/file_index"])
        return p, e[f"videos/{key}/from_timestamp"], e[f"videos/{key}/to_timestamp"]


# --- meta -------------------------------------------------------------------

def cmd_meta(ds, args):
    info = ds.info

    head("meta/info.json")
    keys = ["codebase_version", "robot_type", "fps", "total_episodes", "total_frames",
            "total_tasks", "total_videos", "total_chunks", "chunks_size",
            "data_files_size_in_mb", "video_files_size_in_mb"]
    for k in keys:
        if k in info:
            print(f"{k:24s} {info[k]}")
    print(f"{'data_path':24s} {info['data_path']}")
    print(f"{'video_path':24s} {info['video_path']}")

    head("features（宣言された特徴量と、実体の置き場所）")
    print(f"{'name':32s} {'dtype':8s} {'shape':16s} where")
    for name, f in info["features"].items():
        where = "videos/*.mp4" if f["dtype"] == "video" else "data/*.parquet"
        print(f"{name:32s} {f['dtype']:8s} {str(f['shape']):16s} {where}")

    head("画像特徴量の申告値")
    for key in ds.video_keys():
        print(key)
        for k, v in sorted(ds.info["features"][key].get("info", {}).items()):
            print(f"    {k:24s} {v}")

    head("言語指示")
    for i, t in sorted(ds.tasks().items()):
        print(f"  task_index={i}  {t!r}")

    eps = ds.episodes()
    lengths = sorted(e["length"] for e in eps)
    total = sum(lengths)
    head("エピソード")
    print(f"エピソード数       {len(eps)}")
    print(f"総フレーム数       {total}  (info.json: {ds.info['total_frames']})")
    print(f"長さ min/median/mean/max "
          f"{lengths[0]} / {np.median(lengths):.0f} / {np.mean(lengths):.1f} / {lengths[-1]} フレーム")
    print(f"{'':25s}{lengths[0]/ds.fps:.1f} / {np.median(lengths)/ds.fps:.1f} / "
          f"{np.mean(lengths)/ds.fps:.1f} / {lengths[-1]/ds.fps:.1f} 秒 (fps={ds.fps})")
    print(f"総時間             {total/ds.fps/60:.1f} 分")

    if ds.version == "v3.0":
        head("エピソードはファイル名ではなくメタデータで特定する（v3.0）")
        print(f"{'ep':>4s} {'length':>7s} {'data file':>12s} {'from_index':>11s} {'to_index':>9s}"
              f" {'video file':>12s} {'from_ts':>9s} {'to_ts':>9s}")
        vkey = ds.video_keys()[0]
        for e in eps[:3] + eps[-2:]:
            print(f"{e['episode_index']:4d} {e['length']:7d} "
                  f"{'file-%03d' % e['data/file_index']:>12s} "
                  f"{e['dataset_from_index']:11d} {e['dataset_to_index']:9d} "
                  f"{'file-%03d' % e[f'videos/{vkey}/file_index']:>12s} "
                  f"{e[f'videos/{vkey}/from_timestamp']:9.3f} {e[f'videos/{vkey}/to_timestamp']:9.3f}")
        print("  ※ 先頭3件と末尾2件のみ表示")

    head("統計（正規化に使う値）")
    if ds.version == "v3.0":
        st = load_json(ds.root / "meta" / "stats.json")
        print("meta/stats.json … データセット全体の統計")
        print(f"  特徴量 {len(st)} 個 / action の統計キー: {sorted(st['action'])}")
        for k in ("min", "max", "mean", "std"):
            print(f"  action {k:5s} {np.round(np.array(st['action'][k]).ravel(), 3).tolist()}")
        img = next((k for k in st if k.startswith("observation.images")), None)
        if img:
            print(f"  {img} mean {np.round(np.array(st[img]['mean']).ravel(), 4).tolist()}")
        e0 = ds.episodes()[0]
        sk = sorted({k.split("/")[2] for k in e0 if k.startswith("stats/")})
        print(f"\nmeta/episodes/*.parquet … エピソードごとの統計も同居している")
        print(f"  統計の種類: {sk}")
    else:
        st = load_jsonl(ds.root / "meta" / "episodes_stats.jsonl")
        s0 = st[0]["stats"]
        print(f"meta/episodes_stats.jsonl … 行数 {len(st)}（エピソードごとに1行）")
        print(f"  特徴量 {len(s0)} 個 / action の統計キー: {sorted(s0['action'])}")
        for k in ("min", "max", "mean", "std"):
            print(f"  action {k:5s} {np.round(np.array(s0['action'][k]).ravel(), 3).tolist()}")
        img = next((k for k in s0 if k.startswith("observation.images")), None)
        if img:
            print(f"  {img} mean {np.round(np.array(s0[img]['mean']).ravel(), 4).tolist()}")
        print("  ※ v2.1 に meta/stats.json は無い（読み込み時に集約する）")

    head("ファイルの実数")
    tot_f = tot_b = 0
    for d, n, b in ds.files():
        print(f"{d:8s} {n:4d} files  {b/1024/1024:8.1f} MB")
        tot_f += n
        tot_b += b
    print(f"{'合計':8s} {tot_f:4d} files  {tot_b/1024/1024:8.1f} MB")


# --- data -------------------------------------------------------------------

def cmd_data(ds, args):
    ep = args.episode
    path = ds.parquet_of(ep)

    head(f"data: episode {ep} が入っている parquet")
    print(f"path  {path.relative_to(ds.root)}")
    print(f"size  {path.stat().st_size/1024:.1f} KB")

    table = pq.read_table(path)
    print(f"file 全体  {table.num_rows} 行 / {table.num_columns} 列")

    df = ds.frames(ep)
    if ds.version == "v3.0":
        e = ds.episode(ep)
        print(f"うち episode {ep} の行  {len(df)} 行"
              f"（dataset_from_index {e['dataset_from_index']} 〜 {e['dataset_to_index']}）")
        print("  ← 1 ファイルに複数エピソードが連結されている")

    head("列の構成（画像の列が無いことを確認する）")
    print(f"{'column':24s} {'arrow type':36s} info.json の dtype")
    for f in table.schema:
        declared = ds.info["features"].get(f.name, {}).get("dtype", "-")
        print(f"{f.name:24s} {str(f.type):36s} {declared}")
    img_cols = [f.name for f in table.schema if "image" in f.name]
    print(f"\n画像らしき列: {img_cols if img_cols else 'なし ← 画像は mp4 に別置き'}")

    head(f"episode {ep} の先頭3フレーム")
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
    print(f"fps={ds.fps} → 期待される間隔 {1/ds.fps:.6f} s")
    print(f"実際の間隔 min={d.min():.6f} max={d.max():.6f}")
    print(f"一意な値={np.unique(np.round(d, 9)).tolist()[:5]}")

    head("全 parquet の合計行数（info.json の total_frames と突き合わせる）")
    total = sum(pq.read_metadata(p).num_rows for p in ds.data_files())
    print(f"parquet {len(ds.data_files())} ファイルの合計 {total} 行  /  "
          f"info.json total_frames = {ds.info['total_frames']}  "
          f"→ {'一致' if total == ds.info['total_frames'] else '不一致'}")


# --- video ------------------------------------------------------------------

def cmd_video(ds, args):
    import av

    ep = args.episode
    n_rows = len(ds.frames(ep))

    for key in ds.video_keys():
        path, t0, t1 = ds.video_of(key, ep)
        head(f"video: {key} / episode {ep}")
        print(f"path  {path.relative_to(ds.root)}")
        print(f"size  {path.stat().st_size/1024/1024:.1f} MB")

        with av.open(str(path)) as c:
            v = c.streams.video[0]
            declared = ds.info["features"][key].get("info", {})
            # codec_context.name は「PyAV が選んだデコーダ名」（AV1 なら libdav1d）で、
            # ファイルに書かれているコーデックそのものではない。実体は codec_tag を見る。
            rows = [
                ("codec(tag)", v.codec_context.codec_tag, declared.get("video.codec")),
                ("decoder", v.codec_context.name, "(PyAV が選んだデコーダ)"),
                ("pix_fmt", v.codec_context.pix_fmt, declared.get("video.pix_fmt")),
                ("width", v.codec_context.width, declared.get("video.width")),
                ("height", v.codec_context.height, declared.get("video.height")),
                ("fps", float(v.average_rate), declared.get("video.fps")),
                ("frames", v.frames, None),
                ("duration(s)", round(float(v.duration * v.time_base), 3) if v.duration else None, None),
                ("audio streams", len(c.streams.audio), declared.get("has_audio")),
            ]
            print(f"\n{'項目':14s} {'mp4 の実体':16s} info.json の申告")
            for name, actual, dec in rows:
                print(f"{name:14s} {str(actual):16s} {dec}")
            n_frames = v.frames

        if t0 is None:
            print(f"\nこの mp4 は episode {ep} 専用。"
                  f"\nparquet {n_rows} 行 / mp4 {n_frames} フレーム → "
                  f"{'一致' if n_rows == n_frames else '不一致'}")
        else:
            print(f"\nこの mp4 には複数エピソードが連結されている（v3.0）。")
            print(f"episode {ep} の区間  {t0:.3f} s 〜 {t1:.3f} s  "
                  f"= {(t1-t0)*ds.fps:.0f} フレーム相当")
            print(f"parquet {n_rows} 行 / 区間 {(t1-t0)*ds.fps:.0f} フレーム → "
                  f"{'一致' if abs((t1-t0)*ds.fps - n_rows) < 1 else '不一致'}")
            print(f"（mp4 全体は {n_frames} フレーム）")


# --- diff -------------------------------------------------------------------

def cmd_diff(ds, args):
    ep = args.episode
    df = ds.frames(ep)

    a = np.stack(df["action"].to_numpy())             # [T, 6] リーダーが指令した角度
    s = np.stack(df["observation.state"].to_numpy())  # [T, 6] フォロワーが実際に到達した角度
    names = ds.info["features"]["action"]["names"]
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

    head("追従の遅れ（action を k フレームずらすと state に近づくか）")
    for k in range(0, 5):
        err = np.abs(a[:len(a)-k] - s[k:]).mean()
        print(f"  {k} フレーム遅らせる → 平均絶対差 {err:.3f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(names), 1, figsize=(9, 1.7 * len(names)), sharex=True)
        for i, n in enumerate(names):
            axes[i].plot(ts, a[:, i], label="action", lw=1.2)
            axes[i].plot(ts, s[:, i], label="observation.state", lw=1.2, ls="--")
            axes[i].set_ylabel(n, fontsize=8)
            axes[i].tick_params(labelsize=8)
        axes[0].legend(fontsize=8, ncol=2)
        axes[-1].set_xlabel("timestamp [s]")
        fig.suptitle(f"action vs observation.state (episode {ep})")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"\n図を書き出しました: {args.plot}")


# --- entry ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("meta", "data", "video", "diff", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("root", type=Path)
        sp.add_argument("--episode", type=int, default=0)
        if name in ("diff", "all"):
            sp.add_argument("--plot", default=None, help="PNG の出力先")
    args = p.parse_args()

    ds = DatasetFiles(args.root)
    print(f"codebase_version = {ds.version}  ({ds.root})")

    cmds = {"meta": cmd_meta, "data": cmd_data, "video": cmd_video, "diff": cmd_diff}
    if args.cmd == "all":
        for name in ("meta", "data", "video", "diff"):
            cmds[name](ds, args)
    else:
        cmds[args.cmd](ds, args)


if __name__ == "__main__":
    main()
