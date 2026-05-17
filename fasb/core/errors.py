from __future__ import annotations


class FASBError(Exception):
    """Base framework error."""


class ComponentValidationError(FASBError):
    """Raised when a plugin returns an invalid output."""


class PluginError(FASBError):
    """Raised when plugin execution fails."""


class ConfigurationError(FASBError):
    """Raised when a config cannot be loaded or interpreted."""
