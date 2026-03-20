"""All view frames for the main application window.

Frames are imported lazily by app.py via importlib to speed up startup.
This __init__.py only provides __all__ for discoverability — it does NOT
eagerly import the frame classes.
"""

__all__ = [
    "HomeFrame", "DatasetsFrame", "TableFrame", "AnalyticsFrame",
    "CodingFrame", "AICodingFrame", "TopicModellingFrame",
    "SlicerFrame", "ExportFrame", "SettingsFrame",
]
