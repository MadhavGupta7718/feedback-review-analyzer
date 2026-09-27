"""Theme discovery: MiniLM embeddings -> PCA -> HDBSCAN -> c-TF-IDF keywords -> names + representatives.

Everything is deterministic for a fixed input (PCA uses a fixed seed; HDBSCAN is deterministic).
Theme names are built only from the highest-scoring c-TF-IDF terms of the cluster's own reviews.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from ml import config

PLACEHOLDER_RE = re.compile(r"\[(?:EMAIL|URL|CARD|ORDER_ID|ACCOUNT_ID|CUSTOMER_ID|PHONE|USER|PERSON)\]")
# Generic review words that say nothing about the topic.
DOMAIN_STOPWORDS = {
    "app", "apps", "nimbus", "please", "fix", "asap", "really", "just", "honestly", "seriously", "ugh", "wow",
    "hmm", "okay", "ok", "fyi", "note", "thanks", "thank", "love", "hate", "star", "stars", "update", "updated",
    "terrible", "experience", "disappointed", "happy", "frustrating", "annoying", "uninstalling", "resolved",
    "otherwise", "fine", "hope", "improves", "keep", "good", "great", "work", "lol", "im", "don", "didn", "doesn",
    "ve", "ll", "got", "get", "like", "use", "using", "used", "time", "day", "today", "now", "new", "worst", "ever",
}


@dataclass
class ThemeParams:
    pca_components: int = 20
    # None = scale with batch size: max(15, 0.4% of reviews) -> 40 for a 10K batch. Larger values give
    # cleaner themes but hide small (possibly emerging) issues.
    min_cluster_size: int | None = None
    min_samples: int = 10
    cluster_selection_method: str = "eom"
    # Stage 2: HDBSCAN finds dense micro-clusters (often one per phrasing); micro-clusters whose centroids
    # have cosine similarity >= merge_threshold (average linkage) are merged into one theme.
    merge_threshold: float = 0.65
    noise_assign_threshold: float = 0.60
    n_keywords: int = 8
    n_representatives: int = 5
    seed: int = config.SEED


@dataclass
class Theme:
    theme_id: str
    name: str
    keywords: list[str]
    size: int
    core_size: int
    member_indices: list[int] = field(repr=False, default_factory=list)
    representative_indices: list[int] = field(default_factory=list)
    coherence: float = 0.0  # mean cosine similarity of members to the theme centroid

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("member_indices")
        return d


@dataclass
class ThemeResult:
    themes: list[Theme]
    labels: np.ndarray            # theme index per review, -1 = unassigned
    assignment: list[str]         # "cluster" | "nearest_centroid" | "unassigned"
    similarity: np.ndarray        # cosine similarity to assigned centroid (0 for unassigned)
    stats: dict


def _strip(text: str) -> str:
    return PLACEHOLDER_RE.sub(" ", text)


def embedding_texts(texts: list[str]) -> list[str]:
    """Redaction placeholders carry no topic; left in, "[USER] ..." replies cluster together on the token alone."""
    out = []
    for t in texts:
        s = " ".join(_strip(t).split())
        out.append(s if s else t)
    return out


def ctfidf_keywords(texts: list[str], labels: np.ndarray, n_clusters: int, top_n: int) -> tuple[list[list[str]], list[list[float]]]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    stop = sorted(set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS)
    docs = [" ".join(_strip(texts[i]) for i in np.where(labels == c)[0]) for c in range(n_clusters)]
    vec = CountVectorizer(ngram_range=(1, 2), stop_words=stop, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", min_df=1)
    tf = vec.fit_transform(docs).toarray().astype(float)
    terms = np.array(vec.get_feature_names_out())
    words_per_class = tf.sum(1, keepdims=True)
    avg_words = words_per_class.mean()
    freq = tf.sum(0)
    ctfidf = (tf / np.maximum(words_per_class, 1)) * np.log(1 + avg_words / np.maximum(freq, 1))
    kws, scores = [], []
    for c in range(n_clusters):
        order = np.argsort(-ctfidf[c])[: top_n * 3]
        chosen, sc = [], []
        for j in order:
            t = terms[j]
            if any(t in k.split() or k in t.split() for k in chosen if " " in k or " " in t) and len(chosen) >= 2:
                continue
            chosen.append(t)
            sc.append(float(ctfidf[c, j]))
            if len(chosen) == top_n:
                break
        kws.append(chosen)
        scores.append(sc)
    return kws, scores


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def coverage_names(texts: list[str], labels: np.ndarray, n_clusters: int) -> list[str]:
    """Name = the 1-2 word stems that the largest share of the theme's reviews contain, weighted by lift
    over the whole batch: score = cov_c * log(cov_c / cov_all). Surface form = most frequent variant."""
    from collections import Counter

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop = set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS
    tok = re.compile(r"[a-z][a-z]+")
    doc_stems: list[set[str]] = []
    surface: dict[str, Counter] = {}
    for t in texts:
        stems = set()
        for w in tok.findall(_strip(t).lower()):
            if w in stop:
                continue
            s = _stem(w)
            stems.add(s)
            surface.setdefault(s, Counter())[w] += 1
        doc_stems.append(stems)
    n = len(texts)
    df_all = Counter(s for d in doc_stems for s in d)
    names = []
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        df_c = Counter(s for i in idx for s in doc_stems[i])
        scored = []
        for s, dc in df_c.items():
            cov_c, cov_all = dc / len(idx), df_all[s] / n
            if cov_c < 0.10:
                continue
            scored.append((cov_c * np.log(cov_c / cov_all), cov_c, s))
        scored.sort(reverse=True)
        if not scored:
            names.append("Unnamed Theme")
            continue
        words = [scored[0][2]]
        if len(scored) > 1 and scored[1][0] >= 0.35 * scored[0][0]:
            words.append(scored[1][2])
        names.append(" ".join(surface[s].most_common(1)[0][0] for s in words).title())
    return names


def name_from_keywords(keywords: list[str], scores: list[float]) -> str:
    """Fallback: top bigram if nearly as distinctive as the top term, else the top two unigrams."""
    if not keywords:
        return "Unnamed Theme"
    bigrams = [(k, s) for k, s in zip(keywords, scores) if " " in k]
    if bigrams and bigrams[0][1] >= 0.5 * scores[0]:
        return bigrams[0][0].title()
    unigrams = [k for k in keywords if " " not in k]
    return " ".join(unigrams[:2]).title() if unigrams else keywords[0].title()


def merge_micro_clusters(X: np.ndarray, raw: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage agglomeration of micro-cluster centroids under cosine distance."""
    ids = sorted({c for c in raw if c >= 0})
    if len(ids) < 2 or threshold >= 1.0:
        return raw.copy()
    from sklearn.cluster import AgglomerativeClustering

    cents = np.stack([X[raw == c].mean(0) for c in ids])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12
    agg = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=1.0 - threshold)
    groups = agg.fit_predict(cents)
    mapping = {c: int(g) for c, g in zip(ids, groups)}
    return np.array([mapping.get(c, -1) for c in raw])


def discover(embeddings: np.ndarray, texts: list[str], params: ThemeParams | None = None) -> ThemeResult:
    import hdbscan
    from sklearn.decomposition import PCA

    p = params or ThemeParams()
    n = len(texts)
    X = embeddings
    ncomp = min(p.pca_components, X.shape[1], max(2, n - 1))
    Xr = PCA(n_components=ncomp, random_state=p.seed).fit_transform(X)
    mcs = p.min_cluster_size or max(15, int(round(0.004 * n)))
    p = ThemeParams(**{**asdict(p), "min_cluster_size": mcs})
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min(p.min_samples, mcs),
                                cluster_selection_method=p.cluster_selection_method, metric="euclidean",
                                core_dist_n_jobs=1)
    raw = clusterer.fit_predict(Xr)
    micro = merge_micro_clusters(X, raw, p.merge_threshold)
    cluster_ids = sorted({c for c in micro if c >= 0}, key=lambda c: (-(micro == c).sum(), c))
    remap = {c: i for i, c in enumerate(cluster_ids)}
    labels = np.array([remap.get(c, -1) for c in micro])
    k = len(cluster_ids)
    n_micro = len({c for c in raw if c >= 0})

    centroids = np.zeros((k, X.shape[1]), dtype=np.float32)
    for i in range(k):
        m = X[labels == i].mean(0)
        centroids[i] = m / (np.linalg.norm(m) + 1e-12)

    assignment = ["cluster" if l >= 0 else "unassigned" for l in labels]
    sim = np.zeros(n, dtype=np.float32)
    if k:
        sims = X @ centroids.T
        for idx in range(n):
            if labels[idx] >= 0:
                sim[idx] = sims[idx, labels[idx]]
            else:
                j = int(sims[idx].argmax())
                if sims[idx, j] >= p.noise_assign_threshold:
                    labels[idx] = j
                    assignment[idx] = "nearest_centroid"
                    sim[idx] = sims[idx, j]

    kws, scores = ctfidf_keywords(texts, labels, k, p.n_keywords) if k else ([], [])
    cov_names = coverage_names(texts, labels, k) if k else []
    themes: list[Theme] = []
    used_names: set[str] = set()
    for i in range(k):
        members = np.where(labels == i)[0]
        name = cov_names[i] if cov_names[i] != "Unnamed Theme" else name_from_keywords(kws[i], scores[i])
        if name in used_names:
            extra = next((w for w in kws[i] if w.title() not in name), str(i))
            name = f"{name} / {extra.title()}"
        used_names.add(name)
        order = members[np.argsort(-sim[members])]
        reps, seen = [], set()
        for idx in order:  # most central first, skipping identical texts
            key = texts[idx].lower()
            if key in seen:
                continue
            seen.add(key)
            reps.append(int(idx))
            if len(reps) == p.n_representatives:
                break
        themes.append(Theme(
            theme_id=f"theme_{i + 1:03d}", name=name, keywords=kws[i], size=int(len(members)),
            core_size=int(sum(1 for m in members if assignment[m] == "cluster")),
            member_indices=members.tolist(), representative_indices=reps,
            coherence=round(float(sim[members].mean()), 4),
        ))

    stats = {
        "params": asdict(p),
        "n_reviews": n,
        "n_micro_clusters": n_micro,
        "n_clusters": k,
        "hdbscan_noise_points": int((raw < 0).sum()),
        "hdbscan_noise_fraction": round(float((raw < 0).mean()), 4),
        "reassigned_by_nearest_centroid": int(sum(a == "nearest_centroid" for a in assignment)),
        "unassigned": int((labels < 0).sum()),
        "unassigned_fraction": round(float((labels < 0).mean()), 4),
        "cluster_sizes": [t.size for t in themes],
    }
    return ThemeResult(themes, labels, assignment, sim, stats)


def evaluate_against_truth(labels: np.ndarray, truth: list[str], themes: list[Theme]) -> dict:
    """Only possible on the synthetic dataset, where every review carries a planted theme."""
    from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, normalized_mutual_info_score

    truth_arr = np.array(truth)
    assigned = labels >= 0
    per_theme = []
    for i, t in enumerate(themes):
        vals, counts = np.unique(truth_arr[labels == i], return_counts=True)
        top = int(counts.argmax())
        per_theme.append({"theme_id": t.theme_id, "name": t.name, "size": t.size,
                          "majority_truth": str(vals[top]), "purity": round(float(counts[top] / counts.sum()), 4)})
    # which planted themes were recovered as a majority of some cluster
    recovered = sorted({p["majority_truth"] for p in per_theme})
    return {
        "ari_assigned": round(float(adjusted_rand_score(truth_arr[assigned], labels[assigned])), 4),
        "nmi_assigned": round(float(normalized_mutual_info_score(truth_arr[assigned], labels[assigned])), 4),
        "homogeneity": round(float(homogeneity_score(truth_arr[assigned], labels[assigned])), 4),
        "completeness": round(float(completeness_score(truth_arr[assigned], labels[assigned])), 4),
        "weighted_purity": round(float(sum(p["purity"] * p["size"] for p in per_theme) / max(1, sum(p["size"] for p in per_theme))), 4),
        "planted_themes": sorted(set(truth)),
        "recovered_planted_themes": recovered,
        "missing_planted_themes": sorted(set(truth) - set(recovered)),
        "per_theme": per_theme,
    }
