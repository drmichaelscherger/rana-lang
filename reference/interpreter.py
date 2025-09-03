from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import math

# -----------------------------
# Monoids & helpers
# -----------------------------

class Monoid:
    def identity(self):
        raise NotImplementedError
    def combine(self, a, b):
        raise NotImplementedError

class Sum(Monoid):
    def identity(self): return 0
    def combine(self, a, b): return a + b

class Min(Monoid):
    def identity(self): return math.inf
    def combine(self, a, b): return a if a < b else b

class Max(Monoid):
    def identity(self): return -math.inf
    def combine(self, a, b): return a if a > b else b

@dataclass
class ArgMin(Monoid):
    # stores (val, arg)
    def identity(self): return (math.inf, None)
    def combine(self, a, b):
        return a if a[0] <= b[0] else b

@dataclass
class Count(Monoid):
    def identity(self): return 0
    def combine(self, a, b): return a + b

# -----------------------------
# Core data structures
# -----------------------------

@dataclass
class Relation:
    name: str
    schema: Dict[str, type]
    key_name: str = "key"
    _rows: Dict[Any, Dict[str, Any]] = field(default_factory=dict)

    def insert(self, key: Any, **fields):
        # type sanity (lightweight)
        for f, t in self.schema.items():
            if f not in fields:
                raise KeyError(f"Missing field {f}")
            # (Skip strict type check—prototype)
        self._rows[key] = dict(fields)

    def keys(self) -> Set[Any]:
        return set(self._rows.keys())

    def get(self, key: Any) -> Dict[str, Any]:
        return self._rows[key]

    # Selection → mask
    def select(self, pred: Callable[[Any, Dict[str, Any]], bool]) -> Set[Any]:
        return {k for k, rec in self._rows.items() if pred(k, rec)}

    # Parallel assignment under a mask
    def assign(self, mask: Set[Any], field: str, expr: Callable[[Any, Dict[str, Any]], Any]):
        for k in mask:
            self._rows[k][field] = expr(k, self._rows[k])

    # Scalar reduction under a mask
    def reduce(self, mask: Set[Any], monoid: Monoid, expr: Callable[[Any, Dict[str, Any]], Any]):
        acc = monoid.identity()
        for k in mask:
            acc = monoid.combine(acc, expr(k, self._rows[k]))
        return acc

    # Grouped reduction: produces a new Relation (target) or dict for convenience
    def group_reduce(self,
                     mask: Set[Any],
                     key_fn: Callable[[Any, Dict[str, Any]], Any],
                     aggs: Dict[str, Tuple[Monoid, Callable[[Any, Dict[str, Any]], Any]]]
                    ) -> Dict[Any, Dict[str, Any]]:
        # accs[group_key][field] = accumulated value
        accs: Dict[Any, Dict[str, Any]] = {}
        for k in mask:
            rec = self._rows[k]
            g = key_fn(k, rec)
            if g not in accs:
                accs[g] = {f: mon.identity() for f, (mon, _) in aggs.items()}
            for f, (mon, fn) in aggs.items():
                accs[g][f] = mon.combine(accs[g][f], fn(k, rec))
        return accs

# -----------------------------
# Convenience utilities
# -----------------------------

def mean(sum_val, count_val):
    return sum_val / count_val if count_val else math.nan

# -----------------------------
# Example: k-means in the associative style
# -----------------------------

def kmeans(points: Relation, cents: Relation, iters: int = 10):
    for _ in range(iters):
        # 1) assign nearest centroid via ArgMin
        all_pts = points.keys()
        # Preload centroids as list for speed
        C = list(cents._rows.items())  # [(cid, {x:..., y:...}), ...]
        def nearest(_k, p):
            am = ArgMin()
            best = am.identity()
            for cid, c in C:
                d2 = (p['x'] - c['x'])**2 + (p['y'] - c['y'])**2
                best = am.combine(best, (d2, cid))
            return best[1]
        points.assign(all_pts, 'cluster', nearest)

        # 2) recompute centroids as means of assigned points
        mask = points.keys()
        aggs = {
            'sumx': (Sum(), lambda _k, r: r['x']),
            'sumy': (Sum(), lambda _k, r: r['y']),
            'cnt':  (Count(), lambda _k, r: 1)
        }
        grouped = points.group_reduce(mask, key_fn=lambda _k, r: r['cluster'], aggs=aggs)

        # write back to cents (create if missing)
        newcents = {}
        for cid, acc in grouped.items():
            newcents[cid] = {
                'x': mean(acc['sumx'], acc['cnt']),
                'y': mean(acc['sumy'], acc['cnt'])
            }
        # replace centroids relation with new values (keys preserved from groups)
        cents._rows = newcents

# -----------------------------
# Mini demonstration
# -----------------------------
if __name__ == "__main__":
    # Create a Points relation
    Points = Relation('Points', schema={'x': float, 'y': float, 'cluster': int})
    for i, (x, y) in enumerate([(0.0,0.0),(0.1,0.2),(5.0,5.0),(5.2,4.9),(10.0,10.0)]):
        Points.insert(i, x=x, y=y, cluster=-1)

    # Initial centroids
    Cents = Relation('Cents', schema={'x': float, 'y': float})
    Cents.insert(0, x=0.0, y=0.0)
    Cents.insert(1, x=5.0, y=5.0)

    kmeans(Points, Cents, iters=5)

    print("Centroids:")
    for cid, rec in sorted(Cents._rows.items()):
        print(cid, rec)