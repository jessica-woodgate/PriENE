from src.harvest_exception import NoBerriesException
from src.harvest_exception import OutOfBounds
from src.harvest_exception import NoPathFound
from src.harvest_exception import IllegalBerry
import math
from queue import PriorityQueue

class MovingModule():
    """
    Moving module searches for berries, finds paths to nearest berries, returns coordinates for agent to move to, and handles foraging
    Instance variables:
        agent_id -- unique id of agent
        training -- boolean to indicate if training or testing (if testing, need to check which berries the agent is allocated)
        max_width -- width of grid agent has access to
        max_height -- height of grid agent has access to
        path -- current path to nearest berry
        path_step -- current step along the path
        nearest_berry -- the nearest berry agent
        nearest_berry_coordinates -- coordinates of the nearest berry
    """
    def __init__(self, agent_id, model, training, min_width, max_width, min_height, max_height):
        self.agent_id = agent_id
        self.model = model
        self.training = training
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.path = None
        self.path_step = 0
        self.nearest_berry = None
        self.nearest_berry_coordinates = None
    
    def check_nearest_berry(self, current_pos):
        """
        Check nearest berry finds nearest berry and path towards
        Check coordinates of nearest berry: if a berry has been eaten it will move elsewhere
        If no berries are found, have to wait
        """
        if self.path == None or self.nearest_berry_coordinates != self.nearest_berry.pos:
            self.nearest_berry_coordinates = self._find_nearest_berry_coordinates(current_pos)
            if self.nearest_berry_coordinates == None:
                return False
            if self.training:
                self.nearest_berry = self.model.get_uneaten_berry_by_coords(self.nearest_berry_coordinates)
            else:
                self.nearest_berry = self.model.get_uneaten_berry_by_coords(self.nearest_berry_coordinates, self.agent_id)
            self.path = self._find_path_to_berry(current_pos,self.nearest_berry.pos)
            self.path_step = 0
        return True
    
    def move_towards_berry(self, current_pos):
        """
        Move towards nearest berry if not at the end of the path; check if it has been foraged; if at the end of the path then forage
        """
        if self.nearest_berry.foraged == True:
            return False, current_pos
        else:
            if self.path_step == len(self.path):
                if self._forage(current_pos):
                    self.path = None
                    return True, current_pos
                else:
                    raise NoBerriesException(self.agent_id, current_pos)
            else:
                new_pos = self._move(current_pos, self.path[self.path_step])
                self.path_step += 1
                return False, new_pos
    
    def reset(self):
        """
        Reset current path and nearest berry
        """
        self.path = None
        self.path_step = 0
        self.nearest_berry = None
        self.nearest_berry_coordinates = None
    
    def get_distance_to_berry(self):
        """
        Get the distance to the nearest berry for observations
        """
        if self.path == None:
            return 0
        return len(self.path) - self.path_step
    
    def _move(self, current_pos, action):
        x, y = current_pos
        if action == "north":
            if (y + 1) < self.max_height:
                y += 1
            else:
                raise OutOfBounds(self.agent_id, (x,y+1))
        elif action == "east":
            if (x - 1) >= self.min_width:
                x -= 1
            else:
                raise OutOfBounds(self.agent_id, (x-1,y))
        elif action == "south":
            if (y - 1) >= self.min_height:
                y -= 1
            else:
                raise OutOfBounds(self.agent_id, (x,y-1))
        elif action == "west":
            if (x + 1) < self.max_width:
                x += 1
            else:
                raise OutOfBounds(self.agent_id, (x+1,y))
        return (x, y)
    
    def _forage(self, cell):
        #check if there is a berry at current location
        location = self.model.get_cell_contents(cell)
        for b in location:
            #there can be multiple berries at one location: check we are foraging the one we were going for
            if b.agent_type == "berry" and b.unique_id == self.nearest_berry.unique_id:
                if not self.training and b.allocated_agent_id != self.agent_id:
                    raise IllegalBerry(self.agent_id, f"allocated to agent {b.allocated_agent_id}")
                else:
                    b.foraged = True
                    return True
        return False

    def _calculate_distance(self, point1, point2):
        #Euclidean distance between two points
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def _find_nearest_berry_coordinates(self, agent_coordinates):
        if self.training:
            uneaten_berries_coordinates = self.model.get_uneaten_berries_coordinates()
        else:
            uneaten_berries_coordinates = self.model.get_uneaten_berries_coordinates(self.agent_id)
        if not uneaten_berries_coordinates:
            return None
        #Use the key parameter of min to find the index of the minimum distance
        nearest_berry_index = min(range(len(uneaten_berries_coordinates)), key=lambda i: self._calculate_distance(agent_coordinates, uneaten_berries_coordinates[i]))
        nearest_berry_coordinates = uneaten_berries_coordinates[nearest_berry_index]
        #Return the position of the nearest berry
        return nearest_berry_coordinates

    def _find_path_to_berry(self, agent_coordinates, berry_coordinates):
        def get_neighbours(node):
            x, y = node
            #Possible moves: up, down, right, left
            possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)] 
            return [(new_x, new_y) for dx, dy in possible_moves if self.min_width <= (new_x := x + dx) < self.max_width and self.min_height <= (new_y := y + dy) < self.max_height]
        #stores nodes to be explored; priority is sum of the cost to reach the node ("g_values") and estimated cost to goal
        open_set = PriorityQueue()
        #add start node to priority queue with initial priority of 0
        #agent's coordinates are starting point for exploration and has not incurred any cost so far
        open_set.put((0, agent_coordinates))
        #dictionary that keeps track of parent node for each explored node
        came_from = {}
        #g is the cost to reach the current node from the starting node
        #stores cost of reaching each node from the start node
        g_values = {agent_coordinates: 0}
        #continue until all reachable nodes have been explored/path to goal has been found
        while not open_set.empty():
            #retrieve and remove element with highest priority
            #explore node in order of increasing estimated cost
            current_g, current_node = open_set.get()
            if current_node == berry_coordinates:
                return self._path_to_string(current_node, came_from)
            for neighbour_position in get_neighbours(current_node):
                #calculate tentative g to reach neighbour from current node
                #cost is cost to reach current node + 1 (each step has a uniform cost of 1)
                tentative_g = g_values[current_node] + 1
                #if neighbour not yet been reached or tentative cost is less than known cost to reach neighbour
                if neighbour_position not in g_values or tentative_g < g_values[neighbour_position]:
                    #if true, more optimal path to neighbour has been found, update g_values dictionary, f_value, and came_from
                    g_values[neighbour_position] = tentative_g
                    #f value is lowest estimated cost
                    #calculate estimated total cost to reach goal through current neighbour
                    f_value = tentative_g + self._calculate_distance(neighbour_position, berry_coordinates)
                    open_set.put((f_value, neighbour_position))
                    #optimal path to neighbour_position comes from current node
                    came_from[neighbour_position] = current_node
        raise NoPathFound(self.agent_id, agent_coordinates, berry_coordinates)

    def _path_to_string(self, current_node, came_from):
        path = []
        #continue as long as there are nodes to backtrack in came_from dictionary
        while current_node in came_from:
            next_node = came_from[current_node]
            path.append(self._direction_to_string(current_node, next_node))
            current_node = next_node
        return path[::-1]  # Reverse the path to start from the agent

    def _direction_to_string(self, start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        if dx == 0:
            return "south" if dy > 0 else "north"
        elif dy == 0:
            return "east" if dx > 0 else "west"