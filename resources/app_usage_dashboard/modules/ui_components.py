import streamlit as st
from i18n_setup import _


def update_language():
    import i18n

    i18n.set("locale", st.session_state.language)


def render_global_filters(user_emails):
    # Language selector with on_change
    st.sidebar.radio(
        _("filters.language"),
        options=["en", "ja"],
        index=["en", "ja"].index(st.session_state.get("language", "en")),
        key="language",
        on_change=update_language,
    )

    # Set default values if not present
    if "timeframe_key" not in st.session_state:
        st.session_state["timeframe_key"] = "last_7_days"
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = "ALL"
    if "granularity" not in st.session_state:
        st.session_state["granularity"] = "daily"
    if "date_range" not in st.session_state:
        st.session_state["date_range"] = None

    timeframe_options = [
        ("last_7_days", _("filters.timeframe.last_7_days")),
        ("last_30_days", _("filters.timeframe.last_30_days")),
        ("today", _("filters.timeframe.today")),
        ("yesterday", _("filters.timeframe.yesterday")),
        ("this_month", _("filters.timeframe.this_month")),
        ("last_month", _("filters.timeframe.last_month")),
        ("custom", _("filters.timeframe.custom")),
    ]
    timeframe_keys = [x[0] for x in timeframe_options]
    st.session_state["timeframe_key"] = st.sidebar.selectbox(
        label=_("filters.timeframe.label"),
        options=timeframe_keys,
        format_func=lambda x: dict(timeframe_options)[x],
        key="timeframe_select",
        index=timeframe_keys.index(st.session_state["timeframe_key"]),
    )
    if st.session_state["timeframe_key"] == "custom":
        st.session_state["date_range"] = st.sidebar.date_input(
            _("filters.timeframe.custom"),
            key="custom_date_range",
            value=st.session_state.get("date_range", None),
        )
    else:
        st.session_state["date_range"] = None

    user_email_options = ["ALL"] + user_emails
    st.session_state["user_email"] = st.sidebar.selectbox(
        label=_("filters.user_email.label"),
        options=user_email_options,
        format_func=lambda x: _("filters.user_email.all") if x == "ALL" else x,
        key="user_email_select",
        index=user_email_options.index(st.session_state["user_email"]),
    )

    granularity_options = ["daily", "weekly", "monthly"]
    st.session_state["granularity"] = st.sidebar.selectbox(
        label=_("filters.granularity.label"),
        options=granularity_options,
        format_func=lambda x: _(f"filters.granularity.{x}"),
        key="granularity_select",
        index=granularity_options.index(st.session_state["granularity"]),
    )

    return (
        st.session_state["timeframe_key"],
        st.session_state["date_range"],
        st.session_state["user_email"],
        st.session_state["granularity"],
    )
