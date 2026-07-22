"""EmoC Dashboard — single entry point that unifies every tool on one port.

Normal mode: this Flask server hosts four Dash apps under one URL via
Werkzeug's DispatcherMiddleware, with a tabbed landing page on top:

    /              tabbed shell (nav bar + iframe)
    /viewer/       Brain Viewer        (brain_viewer.py)
    /builder/      RSA Model Builder   (rsa_model_builder.py)
    /scheduler/    RSA Scheduler       (viz/scheduler_app.py)
    /planner/      Analysis Planner    (viz/planner_app.py)

Each sub-app keeps its own Dash instance and callbacks, so there are no
component-ID collisions. To share at a conference, expose port 8050 through a
tunnel (cloudflared / ngrok) — one URL serves every tab.

Launch:  & "C:\\ProgramData\\anaconda3\\python.exe" tools\\dashboard.py
Then open http://127.0.0.1:8050
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # tools/ lives one level below the repo root
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- Mount prefixes: set BEFORE importing the sub-apps -------------------
# Each sub-app reads its own *_URL_BASE at import time and constructs its Dash
# instance with the matching url_base_pathname.
TABS = [
    ("viewer",    "Brain Viewer",    "VIEWER_URL_BASE",    "/viewer/"),
    ("builder",   "RSA Builder",     "BUILDER_URL_BASE",   "/builder/"),
    ("scheduler", "Scheduler",       "SCHEDULER_URL_BASE", "/scheduler/"),
    ("planner",   "Planner",         "PLANNER_URL_BASE",   "/planner/"),
]
for _slug, _label, _env, _base in TABS:
    os.environ[_env] = _base

import flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware

import rsa_model_builder
from viz import viewer_app, scheduler_app, planner_app

SUBAPPS = {
    "/viewer":    viewer_app.app.server,
    "/builder":   rsa_model_builder.app.server,
    "/scheduler": scheduler_app.app.server,
    "/planner":   planner_app.app.server,
}

# --- Parent app: the tabbed landing page ---------------------------------
parent = flask.Flask(__name__)

_NAV = "".join(
    f'<a class="tab" data-src="{base}" href="#{slug}">{label}</a>'
    for slug, label, _env, base in TABS
)

_INDEX_HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EmoC Dashboard</title>
<style>
  html, body {{ margin:0; height:100%; background:#ffffff; color:#222222;
                font-family:'Segoe UI', Arial, sans-serif; }}
  header {{ display:flex; align-items:center; gap:4px; padding:6px 14px;
            background:#f3f5f9; border-bottom:1px solid #d5dbe5; }}
  .brand {{ font-weight:bold; letter-spacing:1px; margin-right:18px; color:#222222; }}
  a.tab {{ color:#445; text-decoration:none; padding:8px 16px; border-radius:6px 6px 0 0;
           font-size:14px; }}
  a.tab:hover {{ background:#e7ecf5; color:#222; }}
  a.tab.active {{ background:#ffffff; color:#4472C4; border:1px solid #d5dbe5; border-bottom:none;
                  font-weight:bold; }}
  iframe {{ border:none; width:100%; height:calc(100vh - 46px); display:block; background:#ffffff; }}
</style>
</head>
<body>
<header>
  <span class="brand">EmoC Dashboard</span>
  {_NAV}
</header>
<iframe id="frame" title="tab"></iframe>
<script>
  const tabs = document.querySelectorAll('a.tab');
  const frame = document.getElementById('frame');
  function activate(slug) {{
    let chosen = null;
    tabs.forEach(t => {{
      const isMatch = t.getAttribute('href') === '#' + slug;
      t.classList.toggle('active', isMatch);
      if (isMatch) chosen = t;
    }});
    if (!chosen) chosen = tabs[0];
    chosen.classList.add('active');
    frame.src = chosen.dataset.src;
  }}
  tabs.forEach(t => t.addEventListener('click', e => {{
    activate(t.getAttribute('href').slice(1));
  }}));
  activate((location.hash || '#viewer').slice(1));
</script>
</body>
</html>
"""


@parent.route("/")
def index():
    return _INDEX_HTML


@parent.route("/healthz")
def healthz():
    return {"status": "ok", "tabs": [slug for slug, *_ in TABS]}


application = DispatcherMiddleware(parent, SUBAPPS)


if __name__ == "__main__":
    from werkzeug.serving import run_simple
    port = int(os.environ.get("DASHBOARD_PORT", "8050"))
    print("=" * 64)
    print("  EmoC Dashboard")
    print(f"  Open http://127.0.0.1:{port}")
    print("  Tabs: " + ", ".join(label for _s, label, _e, _b in TABS))
    print("=" * 64)
    run_simple("0.0.0.0", port, application, use_reloader=False, threaded=True)
