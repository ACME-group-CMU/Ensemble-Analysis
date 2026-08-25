"""Ring statistics for the Si-O-Si network.

Rings are counted on the graph whose nodes are Si and whose edges are Si-O-Si
bridges, following King's shortest-path criterion (King 1967, Franzblau 1991) as
implemented by Le Roux & Jund 2010. Ring size n = number of Si, per Rino 1993.

Two counting conventions are reported:
    RC  unique rings per Si atom  -- the R_N of Rino 1993 and of this project
    RN  ring incidences per Si    -- each ring counted once per Si on it, so RN ~ n*RC

Two-membered rings (edge-sharing Si pairs) are found explicitly, since a
shortest-path search cannot see them.

In a cell this small a path can return to a *different periodic image* of an atom
already on the ring. Such rings are not local features of the network; they are the
cell wrapping onto itself. They are counted separately as `self_image` so the
distinction can be examined rather than silently folded into the totals.

Method attribution:
    King, S. V. Nature 213, 1112 (1967) -- shortest-path ring criterion.
    Franzblau, D. S. Phys. Rev. B 44, 4925 (1991) -- formalization used here.
    Le Roux, S. & Jund, P. Comput. Mater. Sci. 49, 70 (2010) -- ring-finding
        implementation this module follows.
    Rino, J. P. et al. Phys. Rev. B 47, 3053 (1993) -- ring-size convention
        (n = number of Si) and the R_N reference distribution.
"""
from collections import defaultdict, deque

import numpy as np

from src import bonding

MAX_RING_SIZE = 20
IMAGE_RADIUS = 3


def si_graph(structure, cutoff=bonding.SI_O_CUTOFF):
    """Si-Si adjacency through bridging O: {si: [(si_j, relative_image, o_idx), ...]}."""
    table = bonding.neighbor_table(structure, cutoff)
    graph = defaultdict(set)
    for o_idx in bonding.indices(structure, 'O'):
        arms = table.get(o_idx, [])
        for a in range(len(arms)):
            for b in range(len(arms)):
                if a == b:
                    continue
                si_a, img_a, _ = arms[a]
                si_b, img_b, _ = arms[b]
                rel = tuple(int(x) for x in np.array(img_b) - np.array(img_a))
                if si_a == si_b and rel == (0, 0, 0):
                    continue
                graph[si_a].add((si_b, rel, o_idx))
    return {k: sorted(v) for k, v in graph.items()}


def _shortest_path(start, end, graph, focal, max_depth, image_radius):
    """BFS on (base_index, image) nodes, forbidden to pass back through the focal Si."""
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        (base, img), path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for nbr_base, delta, _ in graph.get(base, []):
            nbr_img = tuple(img[k] + delta[k] for k in range(3))
            if any(abs(c) > image_radius for c in nbr_img):
                continue
            node = (nbr_base, nbr_img)
            if node == (focal, (0, 0, 0)) or node in visited:
                continue
            new_path = path + [node]
            if node == end:
                return new_path
            visited.add(node)
            queue.append((node, new_path))
    return None


def _canonical(nodes):
    """Translate a ring so its smallest image is (0,0,0), so copies dedupe."""
    min_img = min(img for _, img in nodes)
    return frozenset((b, tuple(i - m for i, m in zip(img, min_img))) for b, img in nodes)


def _is_self_image(ring):
    """True if the ring visits two different images of the same base atom."""
    bases = [b for b, _ in ring]
    return len(set(bases)) != len(bases)


def find_rings(structure, cutoff=bonding.SI_O_CUTOFF, max_ring_size=MAX_RING_SIZE,
               image_radius=IMAGE_RADIUS):
    """Return {'rings': [(size, canonical_ring, is_self_image)], 'incidences': {si: [...]}}."""
    graph = si_graph(structure, cutoff)
    si_indices = bonding.indices(structure, 'Si')

    rings = {}
    incidences = defaultdict(list)

    # 2-membered rings: Si pairs sharing more than one O, counted once per pair
    for (si_a, si_b, rel), shared in bonding.si_si_shared_oxygens(structure, cutoff).items():
        if len(shared) < 2:
            continue
        ring = _canonical([(si_a, (0, 0, 0)), (si_b, rel)])
        rings[ring] = (2, _is_self_image(ring))
        incidences[si_a].append(2)
        if si_a != si_b:
            incidences[si_b].append(2)

    for focal in si_indices:
        arms = graph.get(focal, [])
        seen_pairs = set()
        for a in range(len(arms)):
            for b in range(a + 1, len(arms)):
                base_a, img_a, o_a = arms[a]
                base_b, img_b, o_b = arms[b]
                if o_a == o_b:
                    continue
                start, end = (base_a, img_a), (base_b, img_b)
                if start == end:
                    continue
                path = _shortest_path(start, end, graph, focal,
                                      max_ring_size - 1, image_radius)
                if path is None:
                    continue
                size = len(path) + 1
                if size > max_ring_size:
                    continue
                nodes = [(focal, (0, 0, 0))] + path
                if len({(bb, ii) for bb, ii in nodes}) != size:
                    continue
                ring = _canonical(nodes)
                if (base_a, img_a, base_b, img_b) not in seen_pairs:
                    seen_pairs.add((base_a, img_a, base_b, img_b))
                    incidences[focal].append(size)
                rings.setdefault(ring, (size, _is_self_image(ring)))

    return {
        'rings': [(size, ring, self_img) for ring, (size, self_img) in rings.items()],
        'incidences': dict(incidences),
        'num_si': len(si_indices),
    }


def ring_statistics(structure, cutoff=bonding.SI_O_CUTOFF, **kwargs):
    """RC and RN by ring size, split by whether the ring closes through a self-image.

    RC_local excludes self-image rings; RC includes them (the legacy behaviour).
    """
    data = find_rings(structure, cutoff, **kwargs)
    n_si = data['num_si']

    rc_all, rc_local, self_img = defaultdict(int), defaultdict(int), defaultdict(int)
    for size, _, is_self in data['rings']:
        rc_all[size] += 1
        if is_self:
            self_img[size] += 1
        else:
            rc_local[size] += 1

    rn = defaultdict(int)
    for sizes in data['incidences'].values():
        for size in sizes:
            rn[size] += 1

    norm = lambda d: {k: v / n_si for k, v in d.items()}
    return {
        'RC': norm(rc_all),
        'RC_local': norm(rc_local),
        'RN': norm(rn),
        'self_image_counts': dict(self_img),
        'num_si': n_si,
        'cutoff': cutoff,
    }
