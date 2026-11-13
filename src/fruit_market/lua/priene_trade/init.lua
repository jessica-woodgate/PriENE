--[[ Melting Pot's trade level (lua/levels/trade/init.lua), with PriENE's read-only observers (priene_components.lua) added. Everything
else, including the settings below, is the trade level's own. ]]

local meltingpot = 'meltingpot.lua.modules.'
local api_factory = require(meltingpot .. 'api_factory')
local avatar_grappling = require(meltingpot .. 'avatar_grappling')
local simulation = require(meltingpot .. 'base_simulation')

-- Required to be able to use the components in the level
local component_library = require(meltingpot .. 'component_library')
local avatar_library = require(meltingpot .. 'avatar_library')
-- Next require the general stamina components
local shared_components = require 'meltingpot.lua.levels.stamina.shared_components'
-- Then the trade level's own components, overriding any previously loaded having the same name
local components = require 'meltingpot.lua.levels.trade.components'
-- Finally PriENE's observers
local priene_components = require 'priene_components'

return api_factory.apiFactory{
    Simulation = simulation.BaseSimulation,
    settings = {
        -- Scale each sprite to a square of size `spriteSize` X `spriteSize`.
        spriteSize = 8,
        -- Terminate the episode after this many frames.
        maxEpisodeLengthFrames = 1000,
        -- Settings to pass to simulation.lua.
        simulation = {},
    }
}
