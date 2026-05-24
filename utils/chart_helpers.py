"""
utils/chart_helpers.py
Shared Plotly layout helpers — consistent 6-month default window
with range selector buttons and pan mode across all pages.
"""
from datetime import datetime, timedelta


def xaxis_range_config(months_back: int = 6) -> dict:
    """
    Returns an xaxis dict that:
      - Defaults the visible window to the last N months
      - Shows 1M / 3M / 6M / 1Y / All range-selector buttons
      - Enables pan by default so the user can scroll back
    Apply to fig.update_layout(xaxis=xaxis_range_config())
    """
    today = datetime.now()
    start = (today - timedelta(days=months_back * 30.5)).strftime("%Y-%m-%d")
    end   = (today + timedelta(days=3)).strftime("%Y-%m-%d")   # tiny right buffer

    return dict(
        range=[start, end],
        type="date",
        rangeslider=dict(visible=False),
        rangeselector=dict(
            buttons=[
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#1e2130",
            activecolor="#00ff88",
            bordercolor="#444",
            font=dict(color="#fafafa", size=11),
            x=0, xanchor="left",
            y=1.04, yanchor="bottom",
        ),
    )


def apply_default_range(fig, months_back: int = 6) -> None:
    """
    Mutates fig in-place:
      - Sets 6-month default view with range selector on the first x-axis
      - Sets dragmode to 'pan' so scroll-back works naturally
    Safe for single-axis and multi-subplot figures.
    """
    cfg = xaxis_range_config(months_back)
    fig.update_layout(
        xaxis=cfg,
        dragmode="pan",
    )
