import sys

import yaml


MODEL_LABELS = {
    "wofost_snomin": "WOFOST SNOMIN",
    "snomin": "WOFOST SNOMIN",
    2: "WOFOST SNOMIN",
    "wofost_cnb": "WOFOST CNB",
    "wofost_classic": "WOFOST CNB",
    "wofost_classic_n": "WOFOST CNB",
    1: "WOFOST CNB",
    "lintul3": "LINTUL3",
    "lintul": "LINTUL3",
    0: "LINTUL3",
}


def print_config_summary(config, stream=None):
    print(format_config_summary(config), file=stream or sys.stderr)


def format_config_summary(config):
    experiment = config["experiment"]
    crop = config["crop"]
    agro = config["crop_model"]["agro"]
    crop_calendar = _load_crop_calendar(agro["path"])
    model = config["crop_model"].get("pcse_model")
    crop_name = crop_calendar.get("crop_name", crop.get("name"))
    variety_name = crop_calendar.get("variety_name", crop.get("variety"))
    start_type = crop_calendar.get("crop_start_type", "unknown")
    if not agro.get("preserve_dates", False) and agro.get("start_type"):
        start_type = agro["start_type"]
    end_type = crop_calendar.get("crop_end_type", "unknown")

    lines = [
        "Resolved experiment configuration",
        f"  Experiment: {experiment.get('name')} (seed={experiment.get('seed')})",
        f"  Crop model: {MODEL_LABELS.get(model, str(model))}",
        f"  Train years: {experiment.get('train_years', [])}",
        f"  Test years: {experiment.get('test_years', [])}",
        f"  Train locations: {_format_locations(experiment.get('train_locations', []))}",
        f"  Test locations: {_format_locations(experiment.get('test_locations', []))}",
        f"  Weather: {_format_weather(config.get('weather', {}))}",
        f"  Environment timestep: {config['environment'].get('timestep', 7)} day(s) per decision",
        f"  Crop: {crop_name} / {variety_name}",
        f"  Crop calendar: start_type={start_type}, end_type={end_type}",
    ]
    return "\n".join(lines)


def _load_crop_calendar(agro_path):
    with open(agro_path, "r") as handle:
        agro = yaml.safe_load(handle)
    if not isinstance(agro, list) or not agro:
        return {}
    first_campaign = agro[0]
    if not isinstance(first_campaign, dict) or not first_campaign:
        return {}
    campaign_body = next(iter(first_campaign.values()))
    if not isinstance(campaign_body, dict):
        return {}
    return campaign_body.get("CropCalendar", {})


def _format_locations(locations):
    formatted = []
    for location in locations:
        if isinstance(location, dict):
            code = location.get("code")
            coordinates = f"({location.get('lat')}, {location.get('lon')})"
            formatted.append(f"{code}={coordinates}" if code else coordinates)
        else:
            formatted.append(str(tuple(location)))
    return ", ".join(formatted) if formatted else "[]"


def _format_weather(weather):
    source = weather.get("source", "provider")
    if source == "file":
        return f"file ({weather.get('file')})"
    if weather.get("random_weather", False):
        return f"random weather ({weather.get('provider') or 'CSV provider'})"
    if source in ("provider", "auto"):
        return f"provider ({weather.get('provider') or 'auto'})"
    return str(source)
