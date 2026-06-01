"""viz — shared building blocks for the EmoC dashboard.

Submodules:
    datasource    Result-folder resolution (Google Drive -> network -> override)
                  and result scanning shared by all dashboard tabs.
    stimuli       EmoC stimulus design constants (labels, colors, partitions).
    scheduler_app Dash app for the Scheduler tab.
    planner_app   Dash app for the Planner tab.
"""

import os


def dash_kwargs(env_var):
    """Dash constructor kwargs for a tab that may be standalone or mounted.

    Standalone (env unset or "/"):     served at the server root.
    Mounted under DispatcherMiddleware: the middleware strips the mount prefix
    before the inner Flask sees the request, so routes must live at "/" while
    browser-facing asset URLs keep the prefix. Hence we split
    requests_pathname_prefix (prefix) from routes_pathname_prefix ("/").
    """
    prefix = os.environ.get(env_var, "/")
    if prefix == "/":
        return {}
    return {"requests_pathname_prefix": prefix, "routes_pathname_prefix": "/"}
