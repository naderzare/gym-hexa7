import gym
from gym import spaces
import numpy as np
import random
import pyglet
pyglet.options["debug_gl"] = False


class DrawText:
    def __init__(self, label: pyglet.text.Label):
        self.label = label

    def render(self):
        self.label.draw()


class Cell:
    def __init__(self, number, neighbours):
        self.number = number
        self.neighbours = neighbours
        self.value = 0

    def __str__(self):
        chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        if self.value == 0:
            if self.number < 9:
                return ' ' + str(self.number) + ' '
            else:
                return str(self.number) + ' '
        return ' ' + chars[self.value - 1] + ' '

    def __repr__(self):
        return str(self.number) + ':' + str(self.value) + ':' + str(self.neighbours)


class Hexa7Env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Hexa7Env, self).__init__()
        self.viewer = None
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(19 + 12,), dtype=np.float16)
        self.graph = [None]
        self.graph.append(Cell(1, [2, 5, 4, 0, 0, 0]))
        self.graph.append(Cell(2, [3, 6, 5, 1, 0, 0]))
        self.graph.append(Cell(3, [0, 7, 6, 2, 0, 0]))
        self.graph.append(Cell(4, [5, 9, 8, 0, 0, 1]))
        self.graph.append(Cell(5, [6, 10, 9, 4, 1, 2]))
        self.graph.append(Cell(6, [7, 11, 10, 5, 2, 3]))
        self.graph.append(Cell(7, [0, 12, 11, 6, 3, 0]))
        self.graph.append(Cell(8, [9, 13, 0, 0, 0, 4]))
        self.graph.append(Cell(9, [10, 14, 13, 8, 4, 5]))
        self.graph.append(Cell(10, [11, 15, 14, 9, 5, 6]))
        self.graph.append(Cell(11, [12, 16, 15, 10, 6, 7]))
        self.graph.append(Cell(12, [0, 0, 16, 11, 7, 0]))
        self.graph.append(Cell(13, [14, 17, 0, 0, 8, 9]))
        self.graph.append(Cell(14, [15, 18, 17, 13, 9, 10]))
        self.graph.append(Cell(15, [16, 19, 18, 14, 10, 11]))
        self.graph.append(Cell(16, [0, 0, 19, 15, 11, 12]))
        self.graph.append(Cell(17, [18, 0, 0, 0, 13, 14]))
        self.graph.append(Cell(18, [19, 0, 0, 17, 14, 15]))
        self.graph.append(Cell(19, [0, 0, 0, 18, 15, 16]))
        self.same_values = []
        self.free_cell = 19
        self.last_generated_tile = list()
        self.set_reward = 0
        self.tile_was_set = False

    def tile_size_available(self):
        for i in range(1, 20):
            if self.graph[i].value > 0:
                continue
            for j in self.graph[i].neighbours:
                if not self.graph[j]:
                    continue
                if self.graph[j].value == 0:
                    return 2
        return 1

    def random_value(self):
        res = np.random.choice(np.arange(1, 7), p=[0.4, 0.3, 0.15, 0.08, 0.05, 0.02])
        return res

    def random_tile(self):
        tile_size = self.tile_size_available()
        if random.random() < 0.3:
            tile_size = 1
        new_tile = list()
        new_tile.append(self.random_value())
        if tile_size == 2:
            new_tile.append(self.random_value())
        new_tile.sort()
        self.last_generated_tile = new_tile
        chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        return [chars[i - 1] for i in new_tile]

    def value_2_char(self, value):
        chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        return chars[value - 1]

    def same_find(self, number, value):
        if self.graph[number].value != value:
            return []
        res = [number]
        self.same_values.append(number)
        for n in self.graph[number].neighbours:
            if not self.graph[n]:
                continue
            if self.graph[n].value != value:
                continue
            if n in self.same_values:
                continue
            r = self.same_find(n, value)
            if r:
                for rr in r:
                    res.append(rr)
                self.same_values.append(n)
        return res

    def find_reward(self, value):
        rewards = [0, 6, 12, 24, 48, 96, 192, 384]
        return rewards[value]

    def state_maker(self):
        obs = np.zeros(self.observation_space.shape)
        for i in range(1, 20):
            cell = self.graph[i]
            obs[i - 1] = cell.value
        for i in range(len(self.last_generated_tile)):
            obs[20 + i * (self.last_generated_tile[i] - 1)] = 1
        return obs

    def step(self, action):
        direction = action % 6
        number = int(action / 6) + 1
        self.tile_was_set = False
        if self.graph[number].value > 0:
            print('here1')
            return self.state_maker(), 0, False, {}
        if len(self.last_generated_tile) == 2:
            if not self.graph[self.graph[number].neighbours[direction]]:
                print('here2')
                return self.state_maker(), 0, False, {}
            if self.graph[self.graph[number].neighbours[direction]].value > 0:
                print('here3')
                return self.state_maker(), 0, False, {}
            cells = [number, self.graph[number].neighbours[direction]]
        else:
            cells = [number]
        for c in range(len(cells)):
            self.graph[cells[c]].value = self.last_generated_tile[c]
        self.set_reward = 0
        c = 0
        if len(cells) == 2:
            if self.graph[cells[0]].value == self.graph[cells[1]].value:
                tmp = cells[0]
                cells[0] = cells[1]
                cells[1] = tmp
        while True:
            if c >= len(cells):
                break
            if self.graph[cells[c]].value == 0:
                c += 1
                continue
            self.same_values.clear()
            value = self.graph[cells[c]].value
            same = self.same_find(cells[c], value)
            if len(same) >= 3:
                self.set_reward += self.find_reward(value)
                for s in same:
                    self.graph[s].value = 0
                if value != 7:
                    self.graph[cells[c]].value = value + 1
                    cells.append(cells[c])
            c += 1
        self.free_cell = 0
        for c in range(1, 20):
            if self.graph[c].value == 0:
                self.free_cell += 1
        self.tile_was_set = True
        self.random_tile()
        print('here4')
        return self.state_maker(), 0, self.finish(), {}
        # information = {'result': 'normal'}
        # if done:
        #     if self.goal_position == self.agent_position:
        #         information['result'] = 'goal'
        #     elif self.agent_position in self.walls:
        #         information['result'] = 'wall'
        #     elif self.current_step > 50:
        #         information['result'] = 'time'
        #     else:
        #         information['result'] = 'out'
        #
        # obs = self._next_observation(done)
        # self.state = obs
        # return obs, reward, done, information

    def reset(self, map_name=None):
        self.same_values = []
        self.free_cell = 19
        self.last_generated_tile = list()
        self.set_reward = 0
        self.tile_was_set = False
        for c in self.graph:
            if c is None:
                continue
            c.value = 0
        self.random_tile()
        return self.state_maker()

    def finish(self):
        if self.free_cell == 0:
            return True
        return False

    def __str__(self):
        l1 = f'    {str(self.graph[17])}  {str(self.graph[18])}  {str(self.graph[19])}\n'
        l2 = f'  {str(self.graph[13])}  {str(self.graph[14])}  {str(self.graph[15])}  {str(self.graph[16])}\n'
        l3 = f'{str(self.graph[8])}  {str(self.graph[9])}  {str(self.graph[10])}  {str(self.graph[11])}  {str(self.graph[12])}\n'
        l4 = f'  {str(self.graph[4])}  {str(self.graph[5])}  {str(self.graph[6])}  {str(self.graph[7])}\n'
        l5 = f'    {str(self.graph[1])}  {str(self.graph[2])}  {str(self.graph[3])}\n'
        tile = f'{[str(self.value_2_char(t)) for t in self.last_generated_tile]}\n'
        return l1 + l2 + l3 + l4 + l5 + tile

    def state(self):
        board = []
        coin = [0, 0]
        for c in range(1, 20):
            board.append(self.graph[c].value)
        for c in range(len(self.last_generated_tile)):
            coin[c] = self.last_generated_tile[c]
        return board, coin

    def reward(self):
        r = 0
        if self.finish():
            r = -10
        elif not self.tile_was_set:
            r = -5
        else:
            r = self.set_reward
            r /= 384.0
            r -= 1.0
        return r

    def __repr__(self):
        return self.__str__()

    def find_color(self, value):
        if value == 0:
            color = (166, 166, 166)
        elif value == 1:
            color = (189, 3, 209)
        elif value == 2:
            color = (255, 137, 0)
        elif value == 3:
            color = (255, 57, 57)
        elif value == 4:
            color = (57, 156, 255)
        elif value == 5:
            color = (85, 255, 85)
        elif value == 6:
            color = (97, 77, 0)
        else:
            color = (255, 247, 0)
        return [c / 255 for c in color]

    def render(self, mode='human', close=False):
        print(self.__str__())
        screen_width = 400
        screen_height = 600
        all_x = [0, 3, 5, 7, 2, 4, 6, 8, 1, 3, 5, 7, 9, 2, 4, 6, 8, 3, 5, 7]
        all_y = [0, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9]
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.cells = []
            self.cells_label = []
            self.tiles = []
            self.tiles_label = []
            for c in self.graph:
                if c is None:
                    continue
                x = all_x[c.number] / 10.0 * 400
                y = all_y[c.number] / 10.0 * 400
                xx = [x - 35, x, x + 35]
                yy = [y - 40 + 200, y - 25 + 200, y + 25 + 200, y + 40 + 200]
                cell = rendering.FilledPolygon([(xx[0], yy[2]),(xx[1], yy[3]),(xx[2], yy[2]),(xx[2], yy[1]),(xx[1], yy[0]),(xx[0], yy[1])])
                cell.set_color(0.5, 0.5, 0.5)
                self.cells.append(cell)
                self.viewer.add_geom(cell)
                number = pyglet.text.Label(
                    str(c.number),
                    font_size=10,
                    bold=False,
                    x=x - 10,
                    y=y + 160,
                    anchor_x="left",
                    anchor_y="bottom",
                    color=(255, 255, 255, 255),
                )
                number.draw()
                self.viewer.add_geom(DrawText(number))
                label = pyglet.text.Label(
                    "",
                    font_size=30,
                    bold=True,
                    x=x - 15,
                    y=y + 170,
                    anchor_x="left",
                    anchor_y="bottom",
                    color=(255, 255, 255, 255),
                )
                label.draw()
                self.viewer.add_geom(DrawText(label))
                self.cells_label.append(label)


            tile1 = rendering.FilledPolygon([(120, 125), (155, 140), (190, 125), (190, 75), (155, 60), (120, 75)])
            self.tiles.append(tile1)
            self.viewer.add_geom(tile1)
            label = pyglet.text.Label(
                "",
                font_size=30,
                bold=True,
                x=138,
                y=75,
                anchor_x="left",
                anchor_y="bottom",
                color=(255, 255, 255, 255),
            )
            label.draw()
            self.viewer.add_geom(DrawText(label))
            self.tiles_label.append(label)
            tile2 = rendering.FilledPolygon([(200, 125), (235, 140), (270, 125), (270, 75), (235, 60), (200, 75)])
            self.tiles.append(tile2)
            self.viewer.add_geom(tile2)
            label = pyglet.text.Label(
                "",
                font_size=30,
                bold=True,
                x=220,
                y=75,
                anchor_x="left",
                anchor_y="bottom",
                color=(255, 255, 255, 255),
            )
            label.draw()
            self.viewer.add_geom(DrawText(label))
            self.tiles_label.append(label)
            self.score_label = pyglet.text.Label(
                "0",
                font_size=30,
                bold=True,
                x=20,
                y=20,
                anchor_x="left",
                anchor_y="bottom",
                color=(0, 0, 0, 255),
            )
            self.score_label.draw()
            self.viewer.add_geom(DrawText(self.score_label))
        for i in range(19):
            cell = self.cells[i]
            color = self.find_color(self.graph[i + 1].value)
            cell.set_color(color[0], color[1], color[2])
            if self.graph[i + 1].value != 0:
                self.cells_label[i].text = str(self.graph[i + 1].value)
            else:
                self.cells_label[i].text = ''
        for i in range(len(self.last_generated_tile)):
            tile = self.tiles[i]
            color = self.find_color(self.last_generated_tile[i])
            tile.set_color(color[0], color[1], color[2])
            self.tiles_label[i].text = str(self.last_generated_tile[i])
        if len(self.last_generated_tile) == 1:
            self.tiles_label[1].text = ""
            self.tiles[1].set_color(166, 166, 166)
        self.score_label.text = str(self.set_reward)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def possible_action(self):
        res = []
        if len(self.last_generated_tile) == 1:
            for i in range(1, 20):
                if self.graph[i].value == 0:
                    for d in range(6):
                        res.append((i - 1) * 6 + d)
        else:
            for i in range(1, 20):
                if self.graph[i].value == 0:
                    for d in range(6):
                        if self.graph[self.graph[i].neighbours[d]] is None:
                            continue
                        if self.graph[self.graph[i].neighbours[d]].value == 0:
                            res.append((i - 1) * 6 + d)
        return res

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
