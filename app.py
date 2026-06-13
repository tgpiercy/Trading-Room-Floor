"""
app.py — StratFlow entrypoint.
Grouped sidebar navigation (st.navigation). Landing page: Market Health.
Sidebar organised by the decision workflow: Orient → Find → Execute → Research.
"""
import streamlit as st

st.set_page_config(page_title="StratFlow", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

import os

_MISSING = []

def _page(path, **kw):
    """st.Page only if the file exists — one missing/misnamed file must
    never take down the whole navigation. Missing paths are collected and
    surfaced as a warning instead of a crash."""
    if not os.path.exists(path):
        _MISSING.append(path)
        return None
    return st.Page(path, **kw)



# ── Grouped sidebar navigation (workflow order) ───────────────────────────────
_nav_spec = {
    "": [_page("pages/6_Market_Health.py", title="Market Health",
               icon="🏥", default=True),
         _page("pages/0_Playbook.py", title="Playbook", icon="📖")],
    "Find & Confirm": [
        _page("pages/2_Flow.py", title="Flow", icon="🌊"),
    ],
    "Execute & Hold": [
        _page("pages/4_Rebalance.py", title="Rebalance", icon="⚖️"),
        _page("pages/3_Portfolio.py", title="Portfolio", icon="💼"),
        _page("pages/12_System_Health.py", title="System Health", icon="🩺"),
    ],
    "Research": [
        _page("pages/9_Backtest.py", title="Backtest", icon="🔬"),
        _page("pages/10_Strategy.py", title="Strategy", icon="⚙️"),
        _page("pages/11_Validation.py", title="Validation", icon="🧪"),
        st.Page("validation_exit_sweep.py", title="Exit Sweep", icon="🚪"),
        st.Page("validation_selection_lab.py", title="Selection Lab", icon="🧬"),
        st.Page("validation_regime_lab.py", title="Regime Lab", icon="🧭"),
        st.Page("validation_hrp_lab.py", title="HRP Lab", icon="🧮"),
    ],
}

_nav_spec = {g: [p for p in pages_ if p is not None]
             for g, pages_ in _nav_spec.items()}
_nav_spec = {g: p for g, p in _nav_spec.items() if p}
if not any(_nav_spec.values()):
    # absolute fallback: never hand st.navigation an empty spec
    def _stub():
        st.title("StratFlow")
        st.error("No page files found in pages/. Upload them to restore navigation.")
    _nav_spec = {"": [st.Page(_stub, title="StratFlow", default=True)]}
if _MISSING:
    st.sidebar.warning("Missing page file(s) skipped:\n" +
                       "\n".join(f"• {m}" for m in _MISSING))
nav = st.navigation(_nav_spec)
nav.run()
