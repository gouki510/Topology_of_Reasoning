def floyd_cycle_detection(graph):
    """
    graph: 各インデックス i に対し、graph[i] が次に遷移するノードのインデックスを表すリスト
    return: (entry, cycle_length)
            entry: ループの入口ノードのインデックス（ループがなければ None）
            cycle_length: ループ内のノード数（すなわち 1 周するのに必要なステップ数）
    """
    slow = 0
    fast = 0
    
    # ループの検出
    while True:
        # 次のノードがリスト外なら終端に達しているとみなす
        if fast >= len(graph) or graph[fast] >= len(graph):
            return None, 0
        slow = graph[slow]              # カメは 1 ステップ進む
        fast = graph[graph[fast]]       # ウサギは 2 ステップ進む
        if slow == fast:
            # ループ発見
            break

    # ループ入口の特定
    slow = 0  # slow をスタート（0）に戻す
    while slow != fast:
        slow = graph[slow]
        fast = graph[fast]
    entry = slow  # これがループの入口

    # ループ長の計算
    cycle_length = 1
    fast = graph[entry]
    while fast != entry:
        fast = graph[fast]
        cycle_length += 1

    return entry, cycle_length

def steps_to_entry(graph, entry):
    """
    スタート（ノード0）からループ入口 entry までのステップ数を計算する。
    """
    steps = 0
    node = 0
    while node != entry:
        node = graph[node]
        steps += 1
    return steps

def count_full_cycles(graph, total_steps):
    """
    graph: 各ノードの遷移を表すリスト
    total_steps: これまでの遷移（ステップ）数 T
    
    return: (loop_exists, full_cycles)
        loop_exists: ループが存在すれば True、存在しなければ False
        full_cycles: ループの部分で完全に回った回数（T がスタートからループ入口に達していない場合は 0）
    """
    entry, cycle_length = floyd_cycle_detection(graph)
    if entry is None:
        # ループが存在しない場合は T 全体が直線状の遷移
        return False, 0

    # ループがある場合、スタートから入口までのステップ数 μ を計算
    mu = steps_to_entry(graph, entry)
    
    # もし総ステップ数が μ 未満なら、ループに入っていない
    if total_steps < mu:
        return True, 0

    # T から入口までの部分を除いた残りがループ内での遷移
    steps_in_cycle = total_steps - mu
    full_cycles = steps_in_cycle // cycle_length
    return True, full_cycles

# --- 使用例 ---
# 例: グラフ [1, 2, 3, 4, 2]
# ノード 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 2 となり、
# ノード 2 からループ（2->3->4->2）が始まり、ループ長は 3 です。
# graph_example = [1, 2, 3, 4, 2]
# total_steps = 20  # 例として 20 ステップ遷移した場合

# loop_exists, full_cycles = count_full_cycles(graph_example, total_steps)
# if loop_exists:
#     print(f"ループは存在します。総 {total_steps} ステップ中、完全なループは {full_cycles} 回起きました。")
# else:
#     print("ループは存在しません。")

from typing import Any, List, Tuple, Dict

def separate_double_and_revision_with_length(
    path: List[Any]
) -> Tuple[List[Dict], List[Dict]]:
    """
    path: ノード遷移の時系列リスト
    戻り値:
      (double_checks, revisions)
      - double_checks: 同じノード v で「同じ次ノード w」だった再訪情報 + 'length'
      - revisions:    同じノード v で「異なる次ノード w」だった再訪情報

    double-check の要素:
      {
        'node':      v,
        'prev_idx':  i,
        'cur_idx':   j,
        'prev_next': w,      # path[i+1]
        'cur_next':  w,      # path[j+1]
        'length':    L       # path[i:i+L] == path[j:j+L]
      }
    revision の要素は以前と同じ形式（length フィールドなし）です。
    """
    last_next: Dict[Any, Any] = {}
    last_idx:  Dict[Any, int] = {}
    double_checks: List[Dict] = []
    revisions: List[Dict] = []
    n = len(path)

    for j in range(n - 1):
        v = path[j]
        w = path[j + 1]

        if v in last_next:
            i      = last_idx[v]
            w_prev = last_next[v]
            info = {
                'node':      v,
                'prev_idx':  i,
                'cur_idx':   j,
                'prev_next': w_prev,
                'cur_next':  w
            }

            if w == w_prev:
                # 同じ次ノードなので "double-check"。ここで連続一致長 L を測る
                L = 0
                while i + L < n and j + L < n and path[i + L] == path[j + L]:
                    L += 1
                info['length'] = L
                double_checks.append(info)
            else:
                # 異なる次ノードなら "revision"
                revisions.append(info)

        # 履歴を更新
        last_next[v] = w
        last_idx[v]  = j

    return double_checks, revisions


import heapq
from collections import defaultdict

def analyze_graph(path, distances):
    """
    path: List[int]
        訪問したノードの軌跡。例: [0, 3, 5, 2, 3, 5, 7]
    distances: List[float]
        path[i] → path[i+1] の距離。len(distances) == len(path)-1

    return: (has_loop, loop_count, diameter)
      - has_loop: bool
            ループがあるなら True（自己ループ＝隣接重複は除く）。
      - loop_count: int
            最初に検出されたループノードの再訪回数（自己ループを除く）。
      - diameter: float
            有向重み付きグラフの直径：
            到達可能な全ノード対 (u,v) の最短経路長の最大値を返す。
            到達不能ペアは無視します。
    """
    # 1) ループ検出（自己ループ除く）
    path = [int(node) for node in path]
    seen = {}
    entry_node = None
    for idx, node in enumerate(path):
        # 隣接重複（自己ループ）は idx - seen[node] == 1 のとき
        if node in seen and idx - seen[node] > 1:
            entry_node = node
            break
        seen[node] = idx

    has_loop = entry_node is not None

    # ループ回数のカウント（自己ループ = 隣接重複はまとめて1回とみなす）
    if has_loop:
        visits = 0
        prev = None
        for node in path:
            # entry_node への訪問で、かつ直前が同じ node でない場合のみカウント
            if node == entry_node and node != prev:
                visits += 1
            prev = node
        loop_count = max(visits - 1, 0)
    else:
        loop_count = 0

    # 2) 有向重み付きグラフの構築
    adj = defaultdict(list)
    for u, v, w in zip(path, path[1:], distances):
        if u == v:
            continue  # 自己ループは無視
        adj[u].append((v, w))

    # 3) 単一始点最短経路（Dijkstra）
    def dijkstra(start):
        dist = {start: 0.0}
        hq = [(0.0, start)]
        while hq:
            d, u = heapq.heappop(hq)
            if d > dist[u]:
                continue
            for v, w in adj.get(u, []):
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(hq, (nd, v))
        return dist

    # 4) 直径の計算
    diameter = 0.0
    for u in adj:
        dist_map = dijkstra(u)
        if len(dist_map) > 1:
            local_max = max(d for node, d in dist_map.items() if node != u)
            diameter = max(diameter, local_max)

    return has_loop, loop_count, diameter

from collections import defaultdict
import heapq

def analyze_graph_v2(path, distances):
    """
    path: List[int]
        訪問したノードの軌跡。例: [0, 3, 5, 2, 3, 5, 7]
    distances: List[float]
        path[i] → path[i+1] の距離。len(distances) == len(path)-1

    return: (has_loop, loop_count, diameter, avg_clustering, avg_path_length, clustering_norm)
      - has_loop: bool
            ループがあるなら True（自己ループ＝隣接重複は除く）。
      - loop_count: int
            最初に検出されたループノードの再訪回数（自己ループを除く）。
      - diameter: float
            有向重み付きグラフの直径。
      - avg_clustering: float
            グラフの平均クラスタ係数（無向化して計算）。
      - avg_path_length: float
            到達可能な全ノード対の最短経路長の平均。
      - clustering_norm: float
            平均クラスタ係数をランダムグラフ基準 C_rand で割った正規化値。
    """
    # --- 1) ループ検出 & カウント ---
    path = [int(node) for node in path]
    seen = {}
    entry_node = None
    for idx, node in enumerate(path):
        if node in seen and idx - seen[node] > 1:
            entry_node = node
            break
        seen[node] = idx

    has_loop = entry_node is not None
    if has_loop:
        visits = 0
        prev = None
        for node in path:
            if node == entry_node and node != prev:
                visits += 1
            prev = node
        loop_count = max(visits - 1, 0)
    else:
        loop_count = 0

    # --- 2) 有向重み付きグラフの構築 ---
    adj = defaultdict(list)
    for u, v, w in zip(path, path[1:], distances):
        if u == v:
            continue
        adj[u].append((v, w))

    # --- 3) Dijkstra 法 ---
    def dijkstra(start):
        dist = {start: 0.0}
        hq = [(0.0, start)]
        while hq:
            d, u = heapq.heappop(hq)
            if d > dist[u]:
                continue
            for v, w in adj.get(u, []):
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(hq, (nd, v))
        return dist

    # --- 4) 直径と平均パス長の計算 ---
    diameter = 0.0
    total_dist = 0.0
    total_pairs = 0
    for u in adj:
        dist_map = dijkstra(u)
        for v, d in dist_map.items():
            if v != u:
                total_dist += d
                total_pairs += 1
        if len(dist_map) > 1:
            local_max = max(d for node, d in dist_map.items() if node != u)
            diameter = max(diameter, local_max)
    avg_path_length = total_dist / total_pairs if total_pairs else 0.0

    # --- 5) クラスタ係数の計算（無向化） ---
    undirected = defaultdict(set)
    for u, vs in adj.items():
        for v, _ in vs:
            undirected[u].add(v)
            undirected[v].add(u)

    clustering_sum = 0.0
    count_cc = 0
    for node, nbrs in undirected.items():
        k = len(nbrs)
        if k < 2:
            continue
        possible = k * (k - 1) / 2
        actual = sum(1 for v in nbrs for w in nbrs if v < w and w in undirected[v])
        clustering_sum += actual / possible
        count_cc += 1
    avg_clustering = clustering_sum / count_cc if count_cc else 0.0

    # --- 6) ランダムグラフ基準との正規化 ---
    N = len(undirected)
    K = sum(len(neighbors) for neighbors in undirected.values()) / N if N else 0.0
    C_rand = (K / (N - 1)) if N > 1 else 0.0
    clustering_norm = avg_clustering / C_rand if C_rand else 0.0

    return has_loop, loop_count, diameter, avg_clustering, avg_path_length, clustering_norm
