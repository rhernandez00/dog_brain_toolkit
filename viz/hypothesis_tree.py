"""viz/hypothesis_tree.py — user-authored hypothesis trees for the EmoC RSA explorer.

Pure logic (no Dash). A *hypothesis tree* is a small nested-dict document the user
authors in the Explorer to map out the RSA hypothesis space. Each node carries:

    id          stable unique string
    label       node name the user types (e.g. "All emotions · dog stimuli")
    link        label for the connector (link) coming in from the parent node
    categories  list of stimulus categories the node represents (free-form;
                the UI seeds them from viz.stimuli but any string is allowed)
    notes       free text
    model       linked RSA model name (folder / csv stem), or None
    children    list of child nodes

The Explorer colours each node by whether its linked model's results exist per
species (dog / human) and, when a node is selected, loads that model's maps into
the 2D comparison panels.

Trees are stored one-per-file as JSON under a per-dataset folder that sits next to
the ``rsa_models`` the nodes point at:

    {datafolder}/{dataset}/hypothesis_trees/{name}.json

so a tree travels with the data and is shared across machines. ``$DBT_TREES_DIR``
overrides the folder; a local ``~/.dbt/hypothesis_trees/{dataset}`` is the offline
fallback.

Run directly for a self-check:
    & "C:\\ProgramData\\anaconda3\\python.exe" viz\\hypothesis_tree.py
"""

import glob
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import datasource  # noqa: E402


# ---------------------------------------------------------------------------
# Node / tree construction
# ---------------------------------------------------------------------------

def _new_id():
    return uuid.uuid4().hex[:8]


def new_node(label="New node", categories=None, notes="", model=None, children=None, link=""):
    return {
        "id": _new_id(),
        "label": label,
        "link": link or "",
        "categories": list(categories) if categories else [],
        "notes": notes or "",
        "model": model,
        "children": list(children) if children else [],
    }


def new_tree(name, dataset, notes="", root_categories=None):
    return {
        "name": name,
        "dataset": dataset,
        "notes": notes or "",
        "root": new_node(label="All categories", categories=root_categories or []),
    }


def ensure_ids(node):
    """Assign ids / fill missing keys in a (possibly hand-edited) tree in place."""
    if not node.get("id"):
        node["id"] = _new_id()
    node.setdefault("label", "Node")
    node.setdefault("link", "")
    node.setdefault("categories", [])
    node.setdefault("notes", "")
    node.setdefault("model", None)
    node.setdefault("children", [])
    for ch in node["children"]:
        ensure_ids(ch)
    return node


# ---------------------------------------------------------------------------
# Traversal / lookup
# ---------------------------------------------------------------------------

def iter_nodes(root, depth=0, parent=None):
    """Yield (node, depth, parent_node) for every node, depth-first."""
    yield root, depth, parent
    for ch in root.get("children", []):
        yield from iter_nodes(ch, depth + 1, root)


def find_node(root, node_id):
    for n, _d, _p in iter_nodes(root):
        if n["id"] == node_id:
            return n
    return None


def find_parent(root, node_id):
    for n, _d, p in iter_nodes(root):
        if n["id"] == node_id:
            return p
    return None


def linked_models(root):
    return {n["model"] for n, _d, _p in iter_nodes(root) if n.get("model")}


# ---------------------------------------------------------------------------
# Mutations (all return True on success)
# ---------------------------------------------------------------------------

def add_child(root, parent_id, node=None):
    parent = find_node(root, parent_id)
    if parent is None:
        return None
    node = node or new_node()
    parent["children"].append(node)
    return node


def add_sibling(root, node_id, node=None):
    parent = find_parent(root, node_id)
    if parent is None:            # node_id is the root — no siblings
        return None
    node = node or new_node()
    idx = next(i for i, c in enumerate(parent["children"]) if c["id"] == node_id)
    parent["children"].insert(idx + 1, node)
    return node


def delete_node(root, node_id):
    parent = find_parent(root, node_id)
    if parent is None:            # cannot delete the root
        return False
    parent["children"] = [c for c in parent["children"] if c["id"] != node_id]
    return True


def move_node(root, node_id, direction):
    """Reorder a node among its siblings. direction: -1 = up/left, +1 = down/right."""
    parent = find_parent(root, node_id)
    if parent is None:
        return False
    kids = parent["children"]
    idx = next((i for i, c in enumerate(kids) if c["id"] == node_id), None)
    if idx is None:
        return False
    new_idx = idx + direction
    if not (0 <= new_idx < len(kids)):
        return False
    kids[idx], kids[new_idx] = kids[new_idx], kids[idx]
    return True


# ---------------------------------------------------------------------------
# Layout — a simple tidy tree (root at depth 0, children below)
# ---------------------------------------------------------------------------

def compute_layout(root):
    """Return {node_id: (x, y)} where y == depth (0 = root) and x spreads leaves
    evenly, each internal node centred over its children. The renderer flips y so
    the root sits at the top with branches descending."""
    pos = {}
    counter = [0]

    def walk(node, depth):
        kids = node.get("children", [])
        if not kids:
            x = float(counter[0])
            counter[0] += 1
        else:
            xs = [walk(ch, depth + 1) for ch in kids]
            x = sum(xs) / len(xs)
        pos[node["id"]] = (x, float(depth))
        return x

    walk(root, 0)
    return pos


# ---------------------------------------------------------------------------
# Result status (per species) for colouring nodes
# ---------------------------------------------------------------------------

def models_with_results(datafolder, dataset, modality, roi):
    """{'D': set(models), 'H': set(models)} — models that have a z / corrected map
    for the given roi (mask type). Empty sets if the disk/roi is unavailable."""
    out = {}
    for sp in ("D", "H"):
        try:
            out[sp] = set(datasource.scan_models(datafolder, dataset, modality, sp, roi))
        except Exception:
            out[sp] = set()
    return out


def node_status(model, result_sets):
    """One of 'both' | 'D' | 'H' | 'none' | 'unlinked' for a node's linked model."""
    if not model:
        return "unlinked"
    has_d = model in result_sets.get("D", set())
    has_h = model in result_sets.get("H", set())
    if has_d and has_h:
        return "both"
    if has_d:
        return "D"
    if has_h:
        return "H"
    return "none"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def trees_dir(datafolder, dataset):
    """Resolve the folder that holds this dataset's tree JSONs (see module docstring)."""
    override = os.environ.get("DBT_TREES_DIR")
    if override:
        return override
    if datafolder and os.path.isdir(datafolder):
        return os.path.join(datafolder, dataset, "hypothesis_trees")
    return os.path.join(os.path.expanduser("~"), ".dbt", "hypothesis_trees", dataset)


def _safe_name(name):
    keep = "-_. "
    cleaned = "".join(c for c in str(name) if c.isalnum() or c in keep).strip()
    return cleaned or "untitled"


def tree_path(dirpath, name):
    return os.path.join(dirpath, f"{_safe_name(name)}.json")


def list_trees(dirpath):
    if not dirpath or not os.path.isdir(dirpath):
        return []
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(dirpath, "*.json")))


def load_tree(dirpath, name):
    with open(tree_path(dirpath, name), "r", encoding="utf-8") as f:
        tree = json.load(f)
    ensure_ids(tree["root"])
    tree.setdefault("name", name)
    tree.setdefault("notes", "")
    return tree


def save_tree(dirpath, tree):
    os.makedirs(dirpath, exist_ok=True)
    ensure_ids(tree["root"])
    path = tree_path(dirpath, tree["name"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def delete_tree(dirpath, name):
    path = tree_path(dirpath, name)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _selfcheck():
    import tempfile

    t = new_tree("check", "EmoC", notes="demo", root_categories=["DgP", "HuP"])
    root_id = t["root"]["id"]
    a = add_child(t["root"], root_id, new_node("Dog stimuli", model="emo-id__dog"))
    b = add_child(t["root"], root_id, new_node("Human stimuli", model="emo-id__hum"))
    add_child(t["root"], a["id"], new_node("Positive only", model="val-bin__dog"))
    add_sibling(t["root"], a["id"], new_node("Cross species", model="emo-id__cross"))

    assert find_node(t["root"], b["id"])["label"] == "Human stimuli"
    assert find_parent(t["root"], a["id"])["id"] == root_id
    assert find_parent(t["root"], root_id) is None
    assert delete_node(t["root"], root_id) is False       # cannot delete root
    n_before = sum(1 for _ in iter_nodes(t["root"]))
    assert move_node(t["root"], b["id"], -1) in (True, False)
    assert linked_models(t["root"]) == {
        "emo-id__dog", "emo-id__hum", "val-bin__dog", "emo-id__cross"}

    pos = compute_layout(t["root"])
    assert pos[root_id][1] == 0.0 and len(pos) == n_before

    assert node_status(None, {}) == "unlinked"
    assert node_status("m", {"D": {"m"}, "H": {"m"}}) == "both"
    assert node_status("m", {"D": {"m"}, "H": set()}) == "D"
    assert node_status("m", {"D": set(), "H": set()}) == "none"

    with tempfile.TemporaryDirectory() as d:
        save_tree(d, t)
        assert list_trees(d) == ["check"]
        t2 = load_tree(d, "check")
        assert linked_models(t2["root"]) == linked_models(t["root"])
        assert delete_tree(d, "check") and list_trees(d) == []

    print("hypothesis_tree self-check: OK")


if __name__ == "__main__":
    _selfcheck()
