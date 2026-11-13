--[[ Read-only observers PriENE adds to Melting Pot's trade level (fruit_market). They only define observations: no updaters, no state
changes and no random draws, so the substrate runs exactly as it does without them. ]]

local args = require 'common.args'
local class = require 'common.class'
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

-- Tree codes in WORLD.TREES (0 is no tree); see src/fruit_market/substrate.py.
local _TREE_CODES = {
    appleTreeHarvestable = 1,
    bananaTreeHarvestable = 2,
    appleTreeUnripe = 3,
    bananaTreeUnripe = 4,
}


--[[ `TreeStateObserver` (on the scene object) adds WORLD.TREES: one code per map cell, indexed [y + 1][x + 1], for every fruit tree's type
and ripeness. ]]
local TreeStateObserver = class.Class(component.Component)

function TreeStateObserver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('TreeStateObserver')},
      {'width', args.positive},
      {'height', args.positive},
  })
  TreeStateObserver.Base.__init__(self, kwargs)
  self._config.width = kwargs.width
  self._config.height = kwargs.height
end

function TreeStateObserver:_trees()
  if not self._treeObjects then
    self._treeObjects = self.gameObject.simulation:getAllGameObjectsWithComponent('FruitType')
  end
  return self._treeObjects
end

function TreeStateObserver:reset()
  self._treeObjects = nil
end

function TreeStateObserver:addObservations(tileSet, world, observations)
  local width = self._config.width
  local height = self._config.height
  observations[#observations + 1] = {
      name = 'WORLD.TREES',
      type = 'tensor.Int32Tensor',
      shape = {height, width},
      func = function(grid)
        local trees = tensor.Int32Tensor(height, width):fill(0)
        for _, tree in ipairs(self:_trees()) do
          local code = _TREE_CODES[tree:getState()]
          if code then
            local position = tree:getPosition()
            trees(position[2] + 1, position[1] + 1):val(code)
          end
        end
        return trees
      end
  }
end


--[[ `HeldByObserver` (on each avatar) adds <index>.HELD_BY: the 1-based index of the player holding this one this step, or 0. ]]
local HeldByObserver = class.Class(component.Component)

function HeldByObserver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('HeldByObserver')},
  })
  HeldByObserver.Base.__init__(self, kwargs)
end

function HeldByObserver:addObservations(tileSet, world, observations)
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  local grappling = self.gameObject:getComponent('Grappling')
  observations[#observations + 1] = {
      name = tostring(playerIndex) .. '.HELD_BY',
      type = 'Doubles',
      shape = {},
      func = function(grid)
        return grappling._heldBy or 0
      end
  }
end


local allComponents = {
    TreeStateObserver = TreeStateObserver,
    HeldByObserver = HeldByObserver,
}

component_registry.registerAllComponents(allComponents)

return allComponents
