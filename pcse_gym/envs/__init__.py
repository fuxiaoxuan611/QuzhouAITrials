__all__ = ["Maize", "WinterWheat"]


def __getattr__(name):
    if name == "Maize":
        from pcse_gym.envs.maize import Maize

        return Maize
    if name == "WinterWheat":
        from pcse_gym.envs.winterwheat import WinterWheat

        return WinterWheat
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
