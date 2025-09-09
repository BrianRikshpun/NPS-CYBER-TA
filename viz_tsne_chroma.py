# viz_tsne_chroma.py
# Visualize a ChromaDB collection with t-SNE (2D/3D), coloring points by source file.
# - Uses consistent colormap for points and legend (no mismatched colors).
# - Works whether embeddings are stored in Chroma or not:
#     * If stored: loads them directly
#     * If not: recomputes using sentence-transformers (MiniLM-L6-v2, 384-dim) just for plotting
# - Robust to scikit-learn versions (n_iter vs max_iter; learning_rate "auto" fallback).
# - Optional auto-discovery: if selected collection has too few items, it can scan disks to find a larger one.

import os
import time
import numpy as np
import chromadb
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple, Optional
from sklearn.manifold import TSNE

# ========= CONFIG (edit as needed) =========
DB_PATH     = r"C:\Users\brian.rikshpun.is\PycharmProjects\NPS-CYBER-TA\chromadb"
COLLECTION  = "nps_cyber_ta-onnx-nomic_embed_text_latest-384"  # set None to auto-pick in this DB
LIMIT       = None           # e.g., 2500 for speed; None = all
PERPLEXITY  = 30.0           # with ~2.5k points, 30–50 works well
LEARNING_RATE = "auto"       # will fall back to 200 if unsupported
N_ITER      = 1000
METRIC      = "euclidean"
PLOT_3D     = True
POINT_SIZE  = 8
ALPHA       = 0.75
MAX_LEGEND_2D = 15
MAX_LEGEND_3D = 12

# If the selected collection has very few items, auto-scan other folders for larger DBs.
ENABLE_AUTO_DISCOVER = True
MIN_ACCEPT_COUNT     = 50
AUTO_SCAN_ROOTS = [
    r"C:\Users\brian.rikshpun.is\PycharmProjects\NPS-CYBER-TA",
    os.path.expanduser("~"),
]
# ==========================================


# ---------- Chroma helpers ----------
def sample_dim(client: chromadb.Client, name: str) -> Optional[int]:
    """Try to sample 1 embedding to infer dimension; may be None if embeddings aren't stored."""
    coll = client.get_collection(name)
    try:
        d = coll.get(include=["embeddings"], limit=1)
        emb_list = d.get("embeddings")
        if not emb_list:
            return None
        e0 = emb_list[0]
        if e0 is None:
            return None
        return len(e0)
    except Exception:
        return None

def coll_count(client: chromadb.Client, name: str) -> int:
    try:
        return client.get_collection(name).count()
    except Exception:
        return -1

def list_collections(client: chromadb.Client) -> List[Tuple[str, int, Optional[int]]]:
    rows = []
    for c in client.list_collections():
        nm = c.name
        cnt = coll_count(client, nm)
        dim = sample_dim(client, nm)  # may be None if not stored
        rows.append((nm, cnt, dim))
    return rows

def choose_best_collection(rows: List[Tuple[str, int, Optional[int]]]) -> Optional[Tuple[str, int, Optional[int]]]:
    """Pick the collection with the largest count; prefer names containing 'nps_cyber_ta'."""
    if not rows:
        return None
    nps = [r for r in rows if "nps_cyber_ta" in r[0].lower()]
    if nps:
        return max(nps, key=lambda r: r[1])
    return max(rows, key=lambda r: r[1])

def scan_for_chroma_dbs(roots: List[str]):
    """Yield (db_path, [(name, count, dim), ...]) for every folder under roots containing chroma.sqlite3."""
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for cur, dirs, files in os.walk(root):
            if "chroma.sqlite3" in files:
                if cur in seen:
                    continue
                seen.add(cur)
                try:
                    client = chromadb.PersistentClient(path=cur)
                    rows = list_collections(client)
                    yield cur, rows
                except Exception:
                    continue

def auto_discover_best(preferred_name: Optional[str] = None):
    """
    Walk AUTO_SCAN_ROOTS for chroma.sqlite3 DBs:
    - If preferred_name is given, pick the DB where that collection exists with the largest count.
    - Else pick the single collection with the largest count overall (preferring 'nps_cyber_ta').
    Returns (db_path, collection_name, count) or (None, None, 0).
    """
    best = (None, None, 0)
    print("[info] Scanning for Chroma DBs...")
    for db_path, rows in scan_for_chroma_dbs(AUTO_SCAN_ROOTS):
        sql = os.path.join(db_path, "chroma.sqlite3")
        sz  = os.path.getsize(sql) if os.path.exists(sql) else 0
        ts  = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(sql))) if os.path.exists(sql) else "n/a"
        print(f"\n== DB: {db_path}  (size={sz:,} bytes, modified={ts}) ==")
        for name, cnt, dim in rows:
            print(f"  {name} -> count={cnt}, dim={dim}")
        if preferred_name:
            same = [r for r in rows if r[0] == preferred_name]
            if same:
                name, cnt, dim = max(same, key=lambda r: r[1])
                if cnt > best[2]:
                    best = (db_path, name, cnt)
        else:
            pick = choose_best_collection(rows)
            if pick and pick[1] > best[2]:
                best = (db_path, pick[0], pick[1])
    return best


# ---------- Loading embeddings/documents ----------
def load_embeddings_or_documents(db_path: str, collection: str, limit: Optional[int]):
    """
    Try to get stored embeddings; if missing, fetch documents and recompute embeddings via sentence-transformers.
    Returns: (X: np.ndarray [N,D], sources: list[str], count_in_collection: int)
    """
    client = chromadb.PersistentClient(path=db_path)
    coll = client.get_collection(collection)
    count_in_coll = coll.count()

    data = coll.get(include=["embeddings", "metadatas", "documents"], limit=limit)
    embs  = data.get("embeddings", None)
    metas = data.get("metadatas") or []
    docs  = data.get("documents") or []

    has_stored = False
    if embs is not None:
        try:
            n = len(embs)
            if n > 0 and embs[0] is not None:
                has_stored = True
        except TypeError:
            has_stored = False

    if has_stored:
        X = np.array(embs, dtype=np.float32)
        sources = [(m.get("source") or m.get("path") or m.get("file") or "unknown") for m in metas]
        return X, sources, count_in_coll

    # No stored embeddings -> recompute from documents
    if not docs:
        raise SystemExit("[error] No embeddings stored AND no documents available to re-embed.\n"
                         f"DB: {db_path}\nCollection: {collection}\n"
                         "Rebuild the index or pick another collection.")
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise SystemExit(
            "[error] This collection doesn’t store embeddings. To visualize, we need to recompute them.\n"
            "Install sentence-transformers:\n"
            "    pip install sentence-transformers\n"
            f"Inner error: {e}"
        )

    print("[info] Computing embeddings on-the-fly with sentence-transformers (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    X = np.asarray(X, dtype=np.float32)
    sources = [(m.get("source") or m.get("path") or m.get("file") or "unknown") for m in metas]
    return X, sources, count_in_coll


# ---------- t-SNE (robust across sklearn versions) ----------
def tsne_reduce(X: np.ndarray, n_components: int, perplexity: float, learning_rate, n_iter: int,
                metric: str, random_state: int = 42):
    kwargs = dict(
        n_components=n_components,
        perplexity=perplexity,
        init="pca",
        metric=metric,
        random_state=random_state,
        verbose=1,
    )

    def _try_with_lr(lr_value):
        kw = dict(kwargs)
        kw["learning_rate"] = lr_value
        try:
            return TSNE(n_iter=n_iter, **kw).fit_transform(X)
        except TypeError:
            return TSNE(max_iter=n_iter, **kw).fit_transform(X)

    try:
        return _try_with_lr(learning_rate)
    except TypeError:
        print(f"[warn] t-SNE learning_rate={learning_rate!r} unsupported; falling back to 200.")
        return _try_with_lr(200)


# ---------- Consistent coloring & plotting ----------
def _resampled_cmap(name: str, n: int):
    try:
        return plt.colormaps.get_cmap(name).resampled(max(n, 1))
    except Exception:
        # Fallback for older Matplotlib
        from matplotlib import cm
        return cm.get_cmap(name, max(n, 1))

def build_color_map(labels: List[str]):
    """Return per-point colors, the sorted class labels, and a stable color_map used by BOTH scatter & legend."""
    classes = sorted(set(os.path.basename(x) for x in labels))
    n = len(classes)
    base = "tab20" if n <= 20 else "hsv"
    cmap = _resampled_cmap(base, n)
    color_map = {lab: cmap(i) for i, lab in enumerate(classes)}
    colors = [color_map[os.path.basename(x)] for x in labels]
    return colors, classes, color_map

def plot_2d(Y2: np.ndarray, sources: List[str], title: str, point_size=POINT_SIZE, alpha=ALPHA, max_legend=MAX_LEGEND_2D):
    colors, classes, color_map = build_color_map(sources)
    plt.figure(figsize=(11, 8))
    plt.scatter(Y2[:, 0], Y2[:, 1], c=colors, s=point_size, alpha=alpha, linewidths=0)
    plt.title(title); plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")

    counts = Counter(os.path.basename(s) for s in sources)
    top = [lab for lab, _ in counts.most_common(max_legend)]
    handles = [
        plt.Line2D([0],[0], marker='o', color='w',
                   label=f"{lab} ({counts[lab]})",
                   markerfacecolor=color_map[lab], markersize=8)
        for lab in top
    ]
    extra = len(classes) - len(top)
    if extra > 0:
        handles.append(plt.Line2D([0],[0], marker='o', color='w',
                                  label=f"... +{extra} more",
                                  markerfacecolor="gray", markersize=8))
    plt.legend(handles=handles, loc="best", frameon=True, fontsize=8)
    plt.tight_layout()

def plot_3d(Y3: np.ndarray, sources: List[str], title: str, point_size=POINT_SIZE, alpha=ALPHA, max_legend=MAX_LEGEND_3D):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    colors, classes, color_map = build_color_map(sources)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Y3[:, 0], Y3[:, 1], Y3[:, 2], c=colors, s=point_size, alpha=alpha, linewidths=0)
    ax.set_title(title); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_zlabel("t-SNE 3")

    counts = Counter(os.path.basename(s) for s in sources)
    top = [lab for lab, _ in counts.most_common(max_legend)]
    handles = [
        plt.Line2D([0],[0], marker='o', color='w',
                   label=f"{lab} ({counts[lab]})",
                   markerfacecolor=color_map[lab], markersize=8)
        for lab in top
    ]
    extra = len(classes) - len(top)
    if extra > 0:
        handles.append(plt.Line2D([0],[0], marker='o', color='w',
                                  label=f"... +{extra} more",
                                  markerfacecolor="gray", markersize=8))
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    plt.tight_layout()


# ---------- Main ----------
def main():
    # 1) Start with configured DB/collection
    db = DB_PATH
    col = COLLECTION

    if not os.path.exists(db):
        raise SystemExit(f"[error] DB_PATH not found: {db}")
    print(f"[info] Using DB_PATH: {db}")
    try:
        print("[info] Folder contents:", ", ".join(os.listdir(db)) or "(empty)")
    except Exception:
        pass

    client = chromadb.PersistentClient(path=db)
    if not col:
        rows = list_collections(client)
        if not rows:
            raise SystemExit("[error] No collections in starting DB; set COLLECTION or rebuild index.")
        # prefer nps_cyber_ta; else largest
        pick = choose_best_collection(rows)
        if not pick:
            raise SystemExit("[error] Could not choose a collection automatically.")
        col = pick[0]
        print(f"[info] Auto-selected collection in starting DB: {col}")

    selected_count = coll_count(client, col)
    print(f"[info] Selected: {col}  (count={selected_count})")

    # 2) If too small, auto-discover a better DB/collection
    if ENABLE_AUTO_DISCOVER and selected_count < MIN_ACCEPT_COUNT:
        print(f"[warn] Collection has only {selected_count} items (< {MIN_ACCEPT_COUNT}). Scanning for a better DB...")
        new_db, new_col, new_cnt = auto_discover_best(preferred_name=col)
        if new_db and new_col and new_cnt > selected_count:
            print(f"[info] Switching to better match:\n"
                  f"      DB:  {new_db}\n"
                  f"      Col: {new_col}  (count={new_cnt})")
            db, col = new_db, new_col

    # 3) Load embeddings (or recompute)
    print(f"[info] Loading from collection: {col}")
    X, sources, final_cnt = load_embeddings_or_documents(db, col, LIMIT)

    # Print top sources (sanity check against legend)
    counts = Counter(os.path.basename(s) for s in sources)
    print("[info] Top sources:", counts.most_common(10))

    n, dim = X.shape
    print(f"[info] Loaded {n} vectors (dim={dim}) from {len(set(sources))} files.")
    if n < max(5.0, PERPLEXITY * 3):
        print(f"[warn] N={n} is small vs perplexity={PERPLEXITY}. Consider lowering PERPLEXITY (e.g., 5–10).")

    # 4) t-SNE + plots
    print("[info] Running t-SNE (2D)…")
    Y2 = tsne_reduce(X, 2, PERPLEXITY, LEARNING_RATE, N_ITER, METRIC, random_state=42)
    plot_2d(Y2, sources, f"t-SNE 2D — {col}")

    if PLOT_3D:
        print("[info] Running t-SNE (3D)…")
        Y3 = tsne_reduce(X, 3, PERPLEXITY, LEARNING_RATE, N_ITER, METRIC, random_state=42)
        plot_3d(Y3, sources, f"t-SNE 3D — {col}")

    print("[ok] Close the plot windows to exit.")
    plt.show()

if __name__ == "__main__":
    main()
