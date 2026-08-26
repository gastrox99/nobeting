---
name: Streamlit widget styling
description: How to reliably scope CSS to a group of Streamlit widgets, especially for responsive overrides.
---

Use a keyed `st.container` to scope CSS to a group of Streamlit widgets. A container key is exposed as an `st-key-...` CSS class, so it is a reliable parent selector for columns and buttons.

**Why:** Streamlit renders widgets as separate frontend blocks; HTML opened and closed in separate Markdown calls cannot act as a wrapper around those widgets. Shared CSS that appears later can also silently override an earlier mobile rule with the same specificity.

**How to apply:** Use `st.container(key=...)` for a styled widget group and target its generated key class. Put mobile-specific rules after shared rules (or use a deliberately higher-specificity selector), and cover both the target dimensions and rule ordering with a regression check.