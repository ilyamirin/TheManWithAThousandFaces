from typing import List
from dataclasses import dataclass
from collections import namedtuple


@dataclass
class TurnoverInput:
    nomenclature: str
    description: str

@dataclass
class BudgetInput:
    obj: str
    project: str
    financing: str

@dataclass
class Prediction:
    value: str
    probability: float

@dataclass
class NetOutput:
    main: Prediction
    alternatives: List[Prediction]
