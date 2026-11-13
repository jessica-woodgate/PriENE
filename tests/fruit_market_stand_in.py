"""
A stand-in for Melting Pot's Fruit Market (src/fruit_market/backend.py's MeltingPotBackend), for testing FruitMarketModel without Melting
Pot installed. It keeps only what the model reads: players' positions, facing, inventories, hunger, offers and rewards, and ripe trees.
Moving forward onto a ripe tree of the player's own fruit harvests 2 of it; eating takes one fruit, resets hunger and gives the substrate's
taste reward; two adjacent players with compatible offers trade what each wants. Nothing is random.
"""
import numpy as np
from src.fruit_market import substrate as fm
from src.fruit_market.state import PlayerState, FruitMarketState, Trade
from src.fruit_market.features import compatible, within_radius

class StandInBackend:
    def __init__(self, roles, trees=None, positions=None, inventories=None):
        self.roles = roles
        self.initial_trees = trees
        self.initial_positions = positions
        self.initial_inventories = inventories
        self.seeds = []
        self.actions = []

    def reset(self, seed):
        self.seeds.append(seed)
        n = len(self.roles)
        positions = self.initial_positions or [(10 + i, 15) for i in range(n)]
        inventories = self.initial_inventories or [(0, 0)] * n
        self.trees = np.zeros((fm.MAP_HEIGHT, fm.MAP_WIDTH), dtype=int) if self.initial_trees is None else self.initial_trees.copy()
        self.hunger = [0.0] * n
        self.players = [PlayerState(i, role, positions[i], 0, tuple(inventories[i]), 0.0, 1.0, 1.0, (0, 0)) for i, role in enumerate(self.roles)]
        return FruitMarketState(list(self.players), self.trees.copy(), [], [0.0] * n)

    def step(self, actions):
        self.actions.append(list(actions))
        rewards = [0.0] * len(self.players)
        players = []
        for p, action in zip(self.players, actions):
            position, inventory, hunger, offer = p.position, list(p.inventory), p.hunger + fm.HUNGER_RATE, p.offer
            if action == 1:
                fx, fy = fm.FORWARD[p.orientation]
                position = (p.position[0] + fx, p.position[1] + fy)
                if self.trees[position[1], position[0]] == fm.RIPE_CODE[p.own_fruit]:
                    inventory[fm.FRUITS.index(p.own_fruit)] += 2
                    self.trees[position[1], position[0]] = fm.UNRIPE_APPLE if p.own_fruit == "apple" else fm.UNRIPE_BANANA
            elif action in fm.EAT_ACTIONS:
                fruit = fm.EAT_ACTIONS[action]
                if inventory[fm.FRUITS.index(fruit)] >= 1:
                    inventory[fm.FRUITS.index(fruit)] -= 1
                    rewards[p.index] = float(fm.taste_value(p.role, fruit))
                    hunger = 0.0
            elif action == fm.CANCEL_ACTION:
                offer = (0, 0)
            elif fm.offer_of(action) is not None:
                offer = fm.offer_of(action)
            #an offer the player can't afford is hidden
            public = offer if all(q >= 0 or held >= -q for q, held in zip(offer, inventory)) else (0, 0)
            players.append(PlayerState(p.index, p.role, position, p.orientation, tuple(inventory), hunger, p.stamina, 1.0, public))
        trades = []
        traded = set()
        for a in players:
            for b in players:
                if a.index < b.index and a.index not in traded and b.index not in traded and compatible(a.offer, b.offer) and within_radius(a.position, b.position):
                    #each gets what it wants, which the other gives (as the substrate's Trading:resolve)
                    a_received = tuple(q if q > 0 else -max(b.offer[f], 0) for f, q in enumerate(a.offer))
                    b_received = tuple(-q for q in a_received)
                    for player, received in [(a, a_received), (b, b_received)]:
                        player.inventory = tuple(held + r for held, r in zip(player.inventory, received))
                        player.offer = (0, 0)
                    trades.append(Trade(a.index, b.index, a_received, b_received))
                    traded |= {a.index, b.index}
        self.players = players
        return FruitMarketState(list(players), self.trees.copy(), trades, rewards)
