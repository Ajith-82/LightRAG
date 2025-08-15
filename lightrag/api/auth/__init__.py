"""
LightRAG Authentication Module.

Enhanced authentication system with password security, rate limiting,
security headers, and comprehensive audit logging.
"""

from .handler import get_auth_handler
from .password_manager import PasswordManager, PasswordPolicy, PasswordStrength

__all__ = ["PasswordManager", "PasswordPolicy", "PasswordStrength", "auth_handler"]


# For backward compatibility
def __getattr__(name):
    if name == "auth_handler":
        return get_auth_handler()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
