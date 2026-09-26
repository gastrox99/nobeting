---
name: Streamlit browser table testing
description: How to read canvas-rendered Streamlit tables reliably in browser regression checks
---

# Canvas tables need focus and rerender awareness
For browser checks of a Streamlit `st.dataframe`, select/copy its Glide grid and
read the tab-separated clipboard text rather than searching for HTML table cells.
Click the grid's scroll surface, focus its canvas, then select all and copy.

**Why:** The table is painted to a canvas; its toolbar's CSV control did not
produce a browser download in this environment. Without focusing the canvas,
select-all sometimes copies the entire page instead. Just after a Streamlit
rerun, the grid may still briefly copy the previous value.

**How to apply:** Read the displayed value through the clipboard, and retry
the assertion for a bounded time after a rerun. Scope the grid to its visible
section so another dataframe cannot satisfy the check.