import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numbers
from PIL import Image
import seaborn as sns
from typing import Callable, List, Tuple, Union

np.set_printoptions(precision=2)

def is_numeric(value):
    return isinstance(value, numbers.Number)

DEBUG = False


class Point():
    p: np.ndarray
    x: int
    y: int

    def __init__(self, p: Union['Point', np.ndarray, float], y: Union[float, None] = None):
        """
        Pass one of the following options as args p, y:
            - Point, None
            - np.ndarray, None
            - float, float
        """
        if y is None:
            if isinstance(p, Point):
                self.p = np.array(p.p, dtype=np.float32)
            else:
                self.p = np.array(p, dtype=np.float32)
        else:
            self.p = np.array([p, y], dtype=np.float32)
        self.x, self.y = self.p[0], self.p[1]

    def __add__(self, p2: 'Point') -> 'Point':
        res = Point(self.p + p2.p)
        return res

    def __sub__(self, p2: 'Point') -> 'Point':
        res = Point(self.p - p2.p)
        return res
    
    def __mul__(self, value: Union['Point', numbers.Number]) -> Union['Point', numbers.Number]:
        if isinstance(value, Point):
            return np.dot(self.p, value.p)
        if isinstance(value, numbers.Number):
            return Point(self.p * value)
        assert True, f"invalid type {type(value)} for multiplying a Point"

    def __rmul__(self, value: Union['Point', numbers.Number]) -> Union['Point', numbers.Number]:
        return self * value

    def norm(self) -> float:
        return np.linalg.norm(self.p)

    def len(self) -> float:
        return self.norm()

    def normalized(self) -> 'Point':
        return Point(self.p / self.norm())
    
    def rotate_90(self) -> 'Point':
        return Point(-self.y, self.x)

    def __str__(self):
        return f'{self.p}'

    def __repr__(self):
        return f'{self.p}'


class Line():
    p1: Point
    p2: Point

    def __init__(self, p1, p2):
        self.p1 = Point(p1)
        self.p2 = Point(p2)

    def __str__(self):
        return f'[{self.p1} - {self.p2}]'

    def __repr__(self):
        return f'[{self.p1} - {self.p2}]'

    def len(self) -> float:
        res = self.p1 - self.p2
        return res.len()

    def point_side(self, p: Point) -> int:
        # Says at which side(kinda left/right) from the line the point is located.
        v_line = self.p2 - self.p1
        v_line_norm = v_line.rotate_90().normalized()
        v_p = p - self.p1

        side_value = v_p * v_line_norm
        return int(np.sign(side_value))
    
    def same_side_points(self, p1: Point, p2: Point) -> bool:
        # Checks if two points are at the same side.
        return False if self.point_side(p1) * self.point_side(p2) == -1 else True

    def intersect_line(self, line2: 'Line') -> bool:
        # Checks if two lines intersect.
        return not (self.same_side_points(line2.p1, line2.p2) or line2.same_side_points(self.p1, self.p2))
    
    def project_point(self, p: Point) -> Point:
        # Finds the closest to p point on the line.
        v_line = self.p2 - self.p1
        v_line_norm = Point(-v_line.y, v_line.x).normalized()
        v_p = p - self.p1

        v_p_projected = v_p - v_line_norm * (v_line_norm * v_p)
        pos_on_line = v_line * v_p_projected
        if pos_on_line <= 0:
            return self.p1  # case: p_projected is located before p1.
        if pos_on_line >= v_line.norm()**2:
            return self.p2  # case: p_projected is located after p2.
        return self.p1 + v_p_projected  # case: p_projected is located between p1 and p2.
    
    def dist_to_point(self, p: Point) -> float:
        projection = self.project_point(p)
        return (p - projection).norm()

    def dist_to_line(self, line: 'Line') -> float:
        d1 = self.dist_to_point(line.p1)
        d2 = self.dist_to_point(line.p2)
        d3 = line.dist_to_point(self.p1)
        d4 = line.dist_to_point(self.p2)
        return np.min(np.array([d1, d2, d3, d4]))

    def reflect_hit(self, ball: 'Ball', direction: Point) -> List[Tuple[Point]]:
        # Returns list of possible reflection from the line. Choose the closest of them.
        projected_center = self.project_point(ball.center)
        path_to_projection = projected_center - ball.center

        options: List[Tuple[Point]] = list()

        normalized_direction = direction.normalized()
        normalized_path_to_projection = path_to_projection.normalized()

        if DEBUG:
            print(f'direction = {direction}')
        if direction * path_to_projection > 0:
            shift = (normalized_path_to_projection * -1) * ball.R
            imaginary_line = Line(self.p1 + shift, self.p2 + shift)

            line_full_path = Line(ball.center, ball.center + direction)
            if imaginary_line.intersect_line(line_full_path):
                if DEBUG:
                    print(f'!!!!! Lines {self} and {line_full_path} DO intersect')
                direction_to_hit = normalized_direction * ((path_to_projection.len() - ball.R) / (normalized_direction * normalized_path_to_projection))
                if direction_to_hit.len() <= direction.len():
                    new_direction = direction - direction_to_hit
                    line_norm = (self.p1 - self.p2).rotate_90().normalized()
                    new_direction -= 2 * line_norm * (line_norm * new_direction)
                    options.append((ball.center + direction_to_hit, new_direction))
            else:
                if DEBUG:
                    print(f'Lines {self} and {line_full_path} don\'t intersect')
        
        for edge in [self.p1, self.p2]:
            path_to_edge = edge - ball.center
            normalized_path_to_edge = path_to_edge.normalized()
            if direction * path_to_edge > 0:
                direction_to_closest: Point = normalized_direction * (normalized_direction * path_to_edge)
                closest_path_to_edge = path_to_edge - direction_to_closest
                if closest_path_to_edge.len() < ball.R:
                    need_gap = np.sqrt(ball.R * ball.R - closest_path_to_edge.len() * closest_path_to_edge.len())
                    direction_to_hit: Point = normalized_direction * (direction_to_closest.len() - need_gap)
                    if direction_to_hit.len() <= direction.len():
                        new_direction = direction - direction_to_hit
                        final_path_to_edge_normalized = (path_to_edge - direction_to_hit).normalized()
                        new_direction -= 2 * final_path_to_edge_normalized * (final_path_to_edge_normalized * new_direction)
                        options.append((ball.center + direction_to_hit, new_direction))

        if not options:
            # If ball doesn't reflect from the line, just let it pass the whole path.
            options.append((ball.center + direction, direction * 0))

        if DEBUG:
            print(f"See options = {options}")
        return options


class Ball():
    center: Point
    R: float

    def __init__(self, center, R):
        self.center = center
        self.R = R

    def __str__(self):
        return f'[{self.center}, {self.R}]'

    def __repr__(self):
        return f'[{self.center}, {self.R}]'

    def intersect_line(self, line: Line) -> bool:
        return line.dist_to_point(self.center) < self.R

    def intersect_ball(self, ball: 'Ball') -> bool:
        return (self.center - ball.center).len() < (self.R + ball.R)
    
    def contains_ball(self, ball: 'Ball') -> bool:
        return (self.center - ball.center).len() < (self.R - ball.R)


def gen_float(_min, _max) -> float:
    return np.random.random() * (_max - _min) + _min

def gen_point(_min, _max) -> Point:
    return Point(gen_float(_min, _max), gen_float(_min, _max))

def gen_line(_min = 0, _max = 100) -> Line:
    return Line(gen_point(_min, _max), gen_point(_min, _max))

def gen_obstacle(min_len = 10, max_len = 30) -> Line:
    line = gen_line()
    line.p2 = line.p1 + (line.p2 - line.p1).normalized() * gen_float(min_len, max_len)
    return line

def conflicting_obstacles(obs1: Line, obs2: Line, confl_dist: float = 3) -> float:
    # If lines intersect or the distance between them is below the threshold they can't be at the same field.
    return obs1.intersect_line(obs2) or obs1.dist_to_line(obs2) < confl_dist


class GolfField():
    field_size: int = 100
    gameball_R: int = 1
    hole_R: int = 3
    obstacles: List[Line] = []

    def __init__(self, seed: int = 0, n_obstacles: int = 15):
        np.random.seed(seed)
        self.obstacles = list()
        self.obstacles.append(Line((0, 0), (0, self.field_size)))
        self.obstacles.append(Line((0, self.field_size), (self.field_size, self.field_size)))
        self.obstacles.append(Line((self.field_size, self.field_size), (self.field_size, 0)))
        self.obstacles.append(Line((self.field_size, 0), (0, 0)))

        for _ in range(n_obstacles):
            self.obstacles.append(self.get_obstacle())
        self.hole = self.get_hole()
        self.gameball = self.get_gameball()

        self.reset()

        if DEBUG:
            print(self.obstacles)

    def correct_obstacle(self, line: Line) -> bool:
        for obstacle in self.obstacles:
            if conflicting_obstacles(obstacle, line):
                return False
        return True
        
    def get_obstacle(self) -> bool:
        new_line = gen_obstacle()
        cnt = 0
        while True:
            if self.correct_obstacle(new_line):
                break
            new_line = gen_obstacle()
            cnt += 1
            assert cnt < 100
        return new_line
        
    def correct_hole(self, hole: Ball) -> bool:
        for obstacle in self.obstacles:
            if hole.intersect_line(obstacle):
                return False
        return True
        
    def get_hole(self):
        while True:
            x, y = np.random.random(2) * self.field_size
            hole = Ball(Point(x, y), self.hole_R)
            if self.correct_hole(hole):
                return hole

    def correct_gameball(self, gameball: Ball) -> bool:
        for obstacle in self.obstacles:
            if gameball.intersect_line(obstacle):
                return False
        if gameball.intersect_ball(self.hole):
            return False
        return True
        
    def get_gameball(self) -> Ball:
        while True:
            x, y = np.random.random(2) * self.field_size
            gameball = Ball(Point(x, y), self.gameball_R)
            if self.correct_gameball(gameball):
                return gameball

    def render(self, user_ax = None, image_path = 'temp_field.png') -> np.ndarray:
        fig, ax = (None, user_ax) if user_ax else plt.subplots(figsize=(7, 7))
        for obstacle in self.obstacles:
            if obstacle.len() == self.field_size:
                ax.plot([obstacle.p1.x, obstacle.p2.x], [obstacle.p1.y, obstacle.p2.y], color='k')
            else:
                ax.plot([obstacle.p1.x, obstacle.p2.x], [obstacle.p1.y, obstacle.p2.y], color='b')

        for i in range(1, len(self.track)):
            if DEBUG:
                print(self.track[i - 1], self.track[i])
            (ball1, col1), (ball2, col2) = self.track[i - 1], self.track[i]
            ax.plot([ball1.center.x, ball2.center.x], [ball1.center.y, ball2.center.y], color=sns.color_palette()[col2 % 10], )


        if self.hole.contains_ball(self.gameball):
            hole = patches.Circle(self.hole.center.p, self.hole.R, edgecolor='g', facecolor='black')
            gameball = patches.Circle(self.gameball.center.p, self.gameball.R, edgecolor='none', facecolor='green')
        else:
            hole = patches.Circle(self.hole.center.p, self.hole.R, edgecolor='r', facecolor='black')
            gameball = patches.Circle(self.gameball.center.p, self.gameball.R, edgecolor='none', facecolor='red')
        ax.add_patch(hole)
        ax.add_patch(gameball)

        plt.savefig(image_path)
        return np.array(Image.open(image_path))

    def render_wind(self, func_for_action: Callable[[float, float], Tuple[float, float]], user_ax = None, image_path = 'temp_wind.png'):
        fig, ax = (None, user_ax) if user_ax else plt.subplots(figsize=(7, 7))
        self.render(ax)

        max_power = 0.1
        for x in range(1, self.field_size, 4):
            for y in range(1, self.field_size, 4):
                angle, power = func_for_action(x, y)
                max_power = max(max_power, power)

        for x in range(1, self.field_size, 4):
            for y in range(1, self.field_size, 4):
                angle, power = func_for_action(x, y)

                colors = [
                    np.array([1.0, 0.0, 0.4]),  # Red 1.3
                    np.array([0.2, 0.3, 1.0]),  # Blue 1.7
                    np.array([0.0, 1.0, 0.3]),  # Green 1.6
                    np.array([0.7, 0.7, 0.1]),  # Yellow 1.2
                ]
                color_i = int(angle // 90)
                color_j = (color_i + 1) % 4
                w_j = (angle % 90) / 90
                w_i = 1 - w_j
                color = w_i * colors[color_i] + w_j * colors[color_j]

                angle *= 2 * np.pi / 360
                direction = Point(np.cos(angle), np.sin(angle)) * 1.5 * (power / max_power)
                ax.arrow(x, y, direction.x, direction.y, head_width=1, head_length=0.5, ec=color)

        plt.savefig(image_path)
        return np.array(Image.open(image_path))


    def move(self, direction: Point) -> None:
        if direction.len() < 1e-4:
            return

        options: List[Tuple[Point]] = list()
        for obstacle in self.obstacles:
            options.extend(obstacle.reflect_hit(self.gameball, direction))

        # TODO: make it a reusable function?
        options = sorted(options, key = lambda x: (x[0] - self.gameball.center).len())  # Choose the first hit to happen.
        assert options
        new_pos, new_direction = options[0]
        if DEBUG:
            print(f'See: {new_pos} {new_direction}')

        self.gameball = Ball(new_pos, self.gameball.R)
        self.track.append((self.gameball, self.current_step))
        self.move(new_direction)

    def reset(self) -> Tuple[Ball, dict]:
        self.gameball = self.get_gameball()
        self.current_step = 0
        self.track = [(self.gameball, self.current_step)]
        info = {
            'track': self.track,
            'current_step': self.current_step,
        }

        return self.gameball, info

    def step(self, action: Tuple[float, float]) -> Tuple[Ball, int, bool, bool, dict]:
        angle, power = action
        assert 0 <= angle < 360, f'angle must be between in range [0, 360)'
        assert 0 < power <= 1e6, f'power must be between in range (0, 1e6]'

        self.current_step += 1

        angle *= 2 * np.pi / 360
        direction = Point(np.cos(angle), np.sin(angle)) * power
        self.move(direction)

        observation = self.gameball
        done = self.hole.contains_ball(self.gameball)
        reward = 1000 if done else 0
        truncated = False
        info = {
            'track': self.track,
            'current_step': self.current_step,
        }

        return observation, reward, done, truncated, info


# Example of creating a Field.
gf = GolfField(4, 15)
