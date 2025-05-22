import os

import i18n
import streamlit as st


def setup_i18n():
    I18N_DIR = os.path.join(os.path.dirname(__file__), "locales")
    i18n.load_path.clear()
    i18n.load_path.append(I18N_DIR)
    i18n.set("filename_format", "{locale}.{format}")
    i18n.set("fallback", "en")
    i18n.set("locale", st.session_state.language)


def _(key, **kwargs):
    return i18n.t(key, **kwargs)
