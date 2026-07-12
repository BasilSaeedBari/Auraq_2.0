"""
Auraq 2.0 — Configuration Manager
Reads and writes settings to %APPDATA%/auraq2/config.ini.
"""
import configparser
import os

_CONFIG_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "auraq2")
CONFIG_PATH = os.path.join(_CONFIG_DIR, "config.ini")

# --------------------------------------------------------------------------- #
# Default values                                                                #
# --------------------------------------------------------------------------- #
_DEFAULTS: dict[str, dict[str, str]] = {
    "General": {
        "download_directory": os.path.join(os.path.expanduser("~"), "Downloads", "Auraq2"),
        "sources_order": "papacambridge,bestexamhelp,dynamicpapers",
        "groq_api_key": "",
        "groq_model": "llama-3.3-70b-versatile",
        "groq_model_fallbacks": "llama-4-scout,openai/gpt-oss-20b,qwen/qwen-3-32b",
        "max_download_workers": "10",
        "max_registry_workers": "4",
        "generate_docx": "no",
        "docx_dpi": "300",
    },
    "Filters": {
        "remove_blank": "yes",
        "remove_additional": "yes",
        "remove_formula": "no",
    },
    "Clipping": {
        "qp_top_margin": "40",
        "qp_bottom_margin": "50",
        "ms_top_margin": "50",
        "ms_bottom_margin": "40",
        "text_end_padding": "8",
    },
    "AI": {
        "batch_confidence_threshold": "0.80",    # raised from 0.70
        "heuristic_fallback_score": "6",
        "strong_heuristic_score": "12",          # heuristic overrides AI below strong_ai_threshold
        "strong_ai_threshold": "0.90",           # AI is unconditionally trusted above this
        "ai_mode": "hybrid",  # "batch", "heuristics", "hybrid"
    },
}


def init_config() -> configparser.ConfigParser:
    """Create config file with defaults if it doesn't exist; merge missing keys otherwise."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    config = configparser.ConfigParser()

    if not os.path.exists(CONFIG_PATH):
        for section, options in _DEFAULTS.items():
            config[section] = options
        _write(config)
    else:
        config.read(CONFIG_PATH, encoding="utf-8")
        updated = False
        for section, options in _DEFAULTS.items():
            if section not in config:
                config[section] = options
                updated = True
            else:
                for key, val in options.items():
                    if key not in config[section]:
                        config[section][key] = val
                        updated = True
        if updated:
            _write(config)

    return config


def load_config() -> configparser.ConfigParser:
    """Load and return the current configuration, initialising if absent."""
    if not os.path.exists(CONFIG_PATH):
        return init_config()
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH, encoding="utf-8")
    return config


def save_config(
    download_dir: str,
    sources: str,
    groq_api_key: str,
    groq_model: str,
    remove_blank: bool,
    remove_additional: bool,
    remove_formula: bool,
    groq_model_fallbacks: str = "llama-4-scout,openai/gpt-oss-20b,qwen/qwen-3-32b",
    ai_mode: str = "hybrid",
    confidence_threshold: float = 0.80,
    heuristic_fallback_score: int = 6,
    strong_ai_threshold: float = 0.90,
) -> None:
    """Persist updated user settings."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    config = load_config()

    config["General"]["download_directory"] = download_dir
    config["General"]["sources_order"] = sources
    config["General"]["groq_api_key"] = groq_api_key
    config["General"]["groq_model"] = groq_model
    config["General"]["groq_model_fallbacks"] = groq_model_fallbacks

    config["Filters"] = {
        "remove_blank": "yes" if remove_blank else "no",
        "remove_additional": "yes" if remove_additional else "no",
        "remove_formula": "yes" if remove_formula else "no",
    }

    if "AI" not in config:
        config["AI"] = {}
    config["AI"]["ai_mode"] = ai_mode
    config["AI"]["batch_confidence_threshold"] = str(confidence_threshold)
    config["AI"]["heuristic_fallback_score"] = str(heuristic_fallback_score)
    config["AI"]["strong_ai_threshold"] = str(strong_ai_threshold)

    _write(config)


def _write(config: configparser.ConfigParser) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        config.write(fh)
