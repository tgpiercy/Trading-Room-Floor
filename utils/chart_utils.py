"""
utils/chart_utils.py
Shared Plotly chart helpers.
"""
import pandas as pd


def set_chart_window(fig, months: int = 6):
    """
    Default the visible window to the last N months.
    All historical data remains loaded — user can pan left or use
    the range-selector buttons to zoom out freely.
    Applies to xaxis (row-1); shared_xaxes links the rest automatically.
    """
    cutoff = (pd.Timestamp.now() - pd.DateOffset(months=months)).strftime("%Y-%m-%d")
    today  = pd.Timestamp.now().strftime("%Y-%m-%d")

    fig.update_layout(
        xaxis=dict(
            range=[cutoff, today],
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M",  step="month", stepmode="backward"),
                    dict(count=3, label="3M",  step="month", stepmode="backward"),
                    dict(count=6, label="6M",  step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year",  stepmode="todate"),
                    dict(count=1, label="1Y",  step="year",  stepmode="backward"),
                    dict(count=2, label="2Y",  step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#1e2130",
                activecolor="#00ff88",
                font=dict(color="#fafafa", size=11),
                bordercolor="#333",
                borderwidth=1,
            ),
            rangeslider=dict(visible=False),
        )
    )
    return fig
