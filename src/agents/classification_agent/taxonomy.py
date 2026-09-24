"""Two label sets the classification agent runs zero-shot against.

SCENARIO_LABELS: MASSIVE's own 18 "scenario" labels (its coarsest intent
grouping). Ground truth exists for these directly in the dataset, so
predictions against this set are scored as real accuracy/F1 in eval.

CUSTOM_LABELS: a support-desk taxonomy this project invented, that the
classifier was never trained on and MASSIVE has no direct field for. There's
no dataset-provided ground truth for it -- SCENARIO_TO_CUSTOM below is a
silver-standard mapping this project authored (deterministic, not model
output) so eval can still report a real accuracy number for "does zero-shot
generalize to an unseen taxonomy" rather than a qualitative shrug.
"""

from __future__ import annotations

SCENARIO_LABELS = [
    "alarm",
    "audio",
    "calendar",
    "cooking",
    "datetime",
    "email",
    "general",
    "iot",
    "lists",
    "music",
    "news",
    "play",
    "qa",
    "recommendation",
    "social",
    "takeaway",
    "transport",
    "weather",
]

CUSTOM_LABELS = [
    "Scheduling & Reminders",
    "Communication",
    "Entertainment & Media",
    "Smart Home & Devices",
    "Information & Lookup",
    "Shopping & Recommendations",
    "Travel & Transport",
]

SCENARIO_TO_CUSTOM = {
    "alarm": "Scheduling & Reminders",
    "calendar": "Scheduling & Reminders",
    "datetime": "Scheduling & Reminders",
    "lists": "Scheduling & Reminders",
    "email": "Communication",
    "social": "Communication",
    "audio": "Entertainment & Media",
    "music": "Entertainment & Media",
    "play": "Entertainment & Media",
    "iot": "Smart Home & Devices",
    "weather": "Information & Lookup",
    "news": "Information & Lookup",
    "qa": "Information & Lookup",
    "general": "Information & Lookup",
    "recommendation": "Shopping & Recommendations",
    "cooking": "Shopping & Recommendations",
    "takeaway": "Shopping & Recommendations",
    "transport": "Travel & Transport",
}

assert set(SCENARIO_TO_CUSTOM) == set(SCENARIO_LABELS)
assert set(SCENARIO_TO_CUSTOM.values()) == set(CUSTOM_LABELS)
