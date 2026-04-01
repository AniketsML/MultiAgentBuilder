import os
import json
import logging

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "agent_prompts.json")

def load_prompt(agent_id: str) -> str:
    """Load the system prompt for a specific agent from the JSON config."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            prompt = data.get(agent_id)
            if not prompt:
                logger.error(f"Prompt for {agent_id} not found in config.")
                return ""
            return prompt
    except Exception as e:
        logger.error(f"Failed to load prompt config for {agent_id}: {e}")
        return ""
