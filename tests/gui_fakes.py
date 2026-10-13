"""Stand-ins the GUI suites hand to the real window."""
import gui

# Keeps ModelLoader's pyqtSignals but never runs (no network, no model load).
NoopLoader = type("NoopLoader", (gui.ModelLoader,), {"start": lambda self: None, "run": lambda self: None})
