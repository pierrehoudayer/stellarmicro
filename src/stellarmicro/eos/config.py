from dataclasses import dataclass

@dataclass(frozen=True)
class EOSOptions:
    debye: bool = True
    radiative: bool = True

