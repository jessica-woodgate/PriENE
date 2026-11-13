from dataclasses import dataclass, field
import numpy as np
from .substrate import ROLE_SPECIALTY, FRUITS, HUNGER_LIMIT, HUNGER_RATE

@dataclass
class PlayerState:
    """
    One player as the backend reads it after a step
        index -- 0-based player index (Melting Pot's index minus 1)
        role -- "apple_farmer" or "banana_farmer"
        position -- (x, y) map cell, y growing southwards
        orientation -- 0 N, 1 E, 2 S, 3 W
        inventory -- (apples, bananas)
        hunger -- HUNGER: 0 just after eating, 1 once hunger costs begin, and more the longer the player stays hungry
        stamina -- STAMINA in [0, 1]
        ready_to_shoot -- READY_TO_SHOOT: 1 if the player may hold
        offer -- the public offer (MY_OFFER) as (apple, banana): (0, 0) if none, or if it can't be afforded
        held_by -- 0-based index of the player holding this one this step, or None
    """
    index: int
    role: str
    position: tuple
    orientation: int
    inventory: tuple
    hunger: float
    stamina: float
    ready_to_shoot: float
    offer: tuple
    held_by: int = None

    @property
    def own_fruit(self):
        return ROLE_SPECIALTY[self.role]

    @property
    def other_fruit(self):
        return FRUITS[1] if self.own_fruit == FRUITS[0] else FRUITS[0]

    def held(self, fruit):
        return self.inventory[FRUITS.index(fruit)]

    @property
    def hungry_steps(self):
        """
        Steps the player has been hungry for (HUNGER at or above the limit; 0 when it isn't)
        """
        return max(round((self.hunger - HUNGER_LIMIT) / HUNGER_RATE), 0) if self.hunger >= HUNGER_LIMIT else 0

@dataclass
class Trade:
    """
    A trade from a Melting Pot "trade" event: player_a gives/receives apples and bananas from player_b (positive = received)
    """
    player_a: int
    player_b: int
    a_received: tuple
    b_received: tuple

@dataclass
class FruitMarketState:
    """
    Everything the scenario reads from the substrate after a step
        players -- PlayerState per player, by index
        trees -- WORLD.TREES: tree code per cell, indexed [y, x] (see substrate.NO_TREE ...)
        trades -- trades completed this step
        rewards -- the substrate's reward for each player this step
    """
    players: list
    trees: np.ndarray
    trades: list = field(default_factory=list)
    rewards: list = field(default_factory=list)

    def trade_of(self, index):
        """
        The trade player index took part in this step (a player trades at most once a step), or None
        """
        for trade in self.trades:
            if index in (trade.player_a, trade.player_b):
                return trade
        return None

    def trade_partner(self, index):
        trade = self.trade_of(index)
        if trade is None:
            return None
        return trade.player_b if trade.player_a == index else trade.player_a

    def received(self, index):
        """
        (apples, bananas) player index received in this step's trade (negative = given); (0, 0) if it didn't trade
        """
        trade = self.trade_of(index)
        if trade is None:
            return (0, 0)
        return trade.a_received if trade.player_a == index else trade.b_received
