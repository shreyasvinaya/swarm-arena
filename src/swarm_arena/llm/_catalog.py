"""Pin LiteLLM's bundled model prices before importing its integrations."""

import os
import sys

_already_local = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"
_already_imported = "litellm" in sys.modules
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

# isort: split
import litellm  # noqa: E402 - the environment pin must precede this import

if _already_imported and not _already_local:
    litellm.model_cost = litellm.get_model_cost_map(url=litellm.model_cost_map_url)
