from collections.abc import Mapping


def aggregate_episode_infos(infos):
    """Merge date-keyed info series while ignoring scalar terminal metadata."""
    episode_info = {}
    for info in infos:
        for key, value in info.items():
            if isinstance(value, Mapping):
                episode_info.setdefault(key, {}).update(value)
    return episode_info
