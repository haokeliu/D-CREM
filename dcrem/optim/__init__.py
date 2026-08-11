"""D-CREM optimisation: Sylvester solver, closed-form updates, trainer."""
from .sylvester import solve_sylvester
from .closed_form import closed_form_W, closed_form_b
from .trainer import DCREMTrainer

__all__ = [
    "solve_sylvester",
    "closed_form_W", "closed_form_b",
    "DCREMTrainer",
]
