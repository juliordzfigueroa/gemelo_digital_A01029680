"""One room of the generator, advancing 5 minutes at a time.

Written like a Mesa agent: it keeps its own state and step() moves it one
tick forward. The simulation will hold one per room and call step() on each,
the way BuildingModel.step() calls every agent.

The state is what carries from one step to the next:

  - the CO2 and the absolute humidity of the equations, which accumulate;
  - whether the lamps are on, the state of the light's Markov chain;
  - the ε of each variable, which is a slow process (generator/noise.py).

Temperature and motion keep no state of their own beyond ε: the first is
read from the hour and the students, the second is drawn fresh each step.

All five variables read the same students N(t), which is what keeps them
related the way the measured ones are.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Section 4.4.2 of the research document.
"""

from __future__ import annotations

import pandas as pd

from common import config as cfg
from generator import co2, humidity, light, motion, noise, temperature

# The variables whose ε is a process carried between steps.
WITH_NOISE = ("co2", "humidity", "temperature", "light")


class RoomGenerator:
    """The five variables of one room, generated step by step."""

    def __init__(self, space_id: int, params: dict, rng) -> None:
        """Start the room empty: CO2 at the outdoor level and lamps off.

        Args:
            space_id: Space identifier.
            params: Parameters of the five variables, as in
                models/generator_params.json.
            rng: A numpy Generator, so a run can be repeated.
        """
        self.space_id = space_id
        self.params = params
        self.rng = rng
        self.co2 = params["co2"]["c_ext"]  # CO2 of the equation, without ε
        self.rho = None  # absolute humidity, set from the outdoor air on the first step
        self.lamps = False
        self.noise = {v: noise.start(params[v]["residual"], rng) for v in WITH_NOISE}

    def step(self, stamp: pd.Timestamp, students: float, weather: pd.Series) -> dict:
        """Advance 5 minutes and return what the sensors would read.

        Args:
            stamp: Start of the interval.
            students: Students in the room during it.
            weather: The weather row of that interval (ingestion.weather).

        Returns:
            One value per variable, in the units of the cleaned data.
        """
        p = self.params
        for variable in WITH_NOISE:
            self.noise[variable] = noise.step(self.noise[variable], p[variable]["residual"], self.rng)

        # Temperature first: the humidity needs it to become relative again.
        t = temperature.expected([stamp], [students], p["temperature"])[0] + self.noise["temperature"]
        t = min(max(t, cfg.PHYSICAL_LIMITS["temperature"][0]), cfg.PHYSICAL_LIMITS["temperature"][1])

        self.co2 = co2.step(self.co2, students, p["co2"])

        outdoor = weather["abs_humidity_ext"]
        self.rho = outdoor if self.rho is None else humidity.step(self.rho, students, outdoor, p["humidity"])
        rh = humidity.relative_humidity(self.rho + self.noise["humidity"], t)

        in_class = students > 0
        sun = weather["radiation_ext"] / 1000
        self.lamps = light.next_state(self.lamps, stamp.hour, in_class, p["light"], self.rng)
        lit = light.expected([self.lamps], [sun], [in_class], p["light"])[0]

        return {
            "co2": co2.reading(self.co2, self.noise["co2"]),
            "temperature": float(t),
            "humidity": float(max(rh, 0.0)),
            "light": light.reading(lit, self.noise["light"]),
            "motion": motion.step(stamp, students, p["motion"], self.rng),
        }
