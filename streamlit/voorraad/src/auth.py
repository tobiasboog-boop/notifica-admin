"""
Simple authentication for Voorraad Dashboard.

Provides password-based access control for the app.
Password is stored in Streamlit secrets (not in code).
"""

import streamlit as st
import hashlib


def hash_password(password: str) -> str:
    """Hash a password for secure storage."""
    return hashlib.sha256(password.encode()).hexdigest()


def check_password() -> bool:
    """
    Returns True if the user has entered the correct password.

    The password hash is stored in .streamlit/secrets.toml:
    [auth]
    password_hash = "..."  # SHA256 hash of the password

    To generate a hash, run:
    python -c "import hashlib; print(hashlib.sha256('YOUR_PASSWORD'.encode()).hexdigest())"
    """

    def password_entered():
        """Check if entered password is correct."""
        entered = st.session_state.get("password", "")
        entered_hash = hash_password(entered)

        # Get expected hash from secrets
        try:
            expected_hash = st.secrets["auth"]["password_hash"]
        except (KeyError, FileNotFoundError):
            # No auth configured - allow access (for development)
            st.session_state["password_correct"] = True
            return

        if entered_hash == expected_hash:
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # Check if already authenticated
    if st.session_state.get("password_correct", False):
        return True

    # Show login form
    st.title("🔐 Voorraad Dashboard")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 8px; text-align: center;">
            <h3>Inloggen</h3>
            <p style="color: #666;">Voer het wachtwoord in om toegang te krijgen tot het dashboard.</p>
        </div>
        """, unsafe_allow_html=True)

        st.text_input(
            "Wachtwoord",
            type="password",
            key="password",
            on_change=password_entered,
            placeholder="Voer wachtwoord in..."
        )

        if st.button("Inloggen", type="primary", use_container_width=True):
            password_entered()
            if st.session_state.get("password_correct", False):
                st.rerun()

        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("❌ Onjuist wachtwoord")

        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; color: #666; font-size: 0.8rem;">
            <p>Neem contact op met Notifica voor toegang.</p>
        </div>
        """, unsafe_allow_html=True)

    return False


def logout():
    """Log out the current user."""
    if "password_correct" in st.session_state:
        del st.session_state["password_correct"]
    st.rerun()
