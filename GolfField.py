import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numbers
from typing import Self, List, Tuple
np.set_printoptions(precision=2)

def is_numeric(value):
    return isinstance(value, numbers.Number)

class Point():
    p: np.ndarray
    x: int
    y: int

    def __init__(self, p: np.ndarray, y = None):
        if y is None:
            if isinstance(p, Point):
                self.p = np.array(p.p)
            else:
                self.p = np.array(p)
        else:
            self.p = np.array([p, y])
        self.x, self.y = self.p[0], self.p[1]

    def __add__(self, p2):
        res = Point(self.p + p2.p)
        return res

    def __sub__(self, p2):
        res = Point(self.p - p2.p)
        return res
    
    def __mul__(self, value):
        if isinstance(value, Point):
            return np.dot(self.p, value.p)
        if isinstance(value, numbers.Number):
            return Point(self.p * value)
        assert True, f"invalid type {type(value)} for multiplying a Point"

    def __rmul__(self, scalar):
        return self * scalar

    def norm(self):
        return np.linalg.norm(self.p)

    def len(self):
        return self.norm()

    def normalized(self):
        return Point(self.p / self.norm())
    
    def rotate_90(self):
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

    def len(self):
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

    def intersect_line(self, line2: Self) -> bool:
        # Checks if two lines intersect
        return not (self.same_side_points(line2.p1, line2.p2) or line2.same_side_points(self.p1, self.p2))
    
    def project_point(self, p: Point) -> Point:
        # Find the closest point on the line to p.
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
    
    def dist_to_point(self, p: Point):
        projection = self.project_point(p)
        return (p - projection).norm()

    def dist_to_line(self, line: Self):
        d1 = self.dist_to_point(line.p1)
        d2 = self.dist_to_point(line.p2)
        d3 = line.dist_to_point(self.p1)
        d4 = line.dist_to_point(self.p2)
        # print(f"\t\t Dist({self}, {line}) = {np.min(np.array([d1, d2, d3, d4]))}")
        return np.min(np.array([d1, d2, d3, d4]))

    def reflect_hit(self, ball: 'Ball', direction: Point) -> List[Tuple[Point]]:
        projected_center = self.project_point(ball.center)
        path_to_projection = projected_center - ball.center

        options: List[Tuple[Point]] = list()

        normalized_direction = direction.normalized()
        normalized_path_to_projection = path_to_projection.normalized()


        print(f'direction = {direction}')
        if direction * path_to_projection > 0:
            shift = (normalized_path_to_projection * -1) * ball.R
            imaginary_line = Line(self.p1 + shift, self.p2 + shift)

            line_full_path = Line(ball.center, ball.center + direction)
            if imaginary_line.intersect_line(line_full_path):
                print(f'!!!!! Lines {self} and {line_full_path} DO intersect')
                direction_to_hit = normalized_direction * ((path_to_projection.len() - ball.R) / (normalized_direction * normalized_path_to_projection))
                if direction_to_hit.len() <= direction.len():                    
                    new_direction = direction - direction_to_hit
                    line_norm = (self.p1 - self.p2).rotate_90().normalized()
                    new_direction -= 2 * line_norm * (line_norm * new_direction)
                    options.append((ball.center + direction_to_hit, new_direction))
            else:
                print(f'Lines {self} and {line_full_path} don\'t intersect')
        
        if not options:
            options.append((ball.center + direction, direction * 0))

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

    def intersect_ball(self, ball: Self) -> bool:
        return (self.center - ball.center).len() < (self.R + ball.R)
    
    def contains_ball(self, ball: Self) -> bool:
        return (self.center - ball.center).len() < (self.R - ball.R)


def gen_float(_min, _max):
    return np.random.random() * (_max - _min) + _min

def gen_point(_min, _max) -> Point:
    return Point(gen_float(_min, _max), gen_float(_min, _max))

def gen_line(_min = 0, _max = 100) -> Line:
    return Line(gen_point(_min, _max), gen_point(_min, _max))

def gen_obstacle(min_len=10, max_len=30) -> Line:
    line = gen_line()
    line.p2 = line.p1 + (line.p2 - line.p1).normalized() * gen_float(min_len, max_len)
    return line

def conflicting_obstacles(obs1: Line, obs2: Line):
    return obs1.intersect_line(obs2) or obs1.dist_to_line(obs2) < 3

class GolfField():
    gameball_R: int = 1
    hole_R: int = 3
    obstacles: List[Line] = []

    def __init__(self, seed = 0, n_obstacles = 15):
        np.random.seed(seed)
        self.obstacles = list()
        self.obstacles.append(Line((0, 0), (0, 100)))
        self.obstacles.append(Line((0, 100), (100, 100)))
        self.obstacles.append(Line((100, 100), (100, 0)))
        self.obstacles.append(Line((100, 0), (0, 0)))

        for _ in range(n_obstacles):
            self.obstacles.append(self.get_obstacle())
        self.hole = self.get_hole()
        self.gameball = self.get_gameball()

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
            x, y = np.random.random(2) * 100
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
        
    def get_gameball(self):
        while True:
            x, y = np.random.random(2) * 100
            gameball = Ball(Point(x, y), self.gameball_R)
            if self.correct_gameball(gameball):
                return gameball

    def plot(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        for obstacle in self.obstacles:
            if obstacle.len() == 100:
                ax.plot([obstacle.p1.x, obstacle.p2.x], [obstacle.p1.y, obstacle.p2.y], color='k')
            else:
                ax.plot([obstacle.p1.x, obstacle.p2.x], [obstacle.p1.y, obstacle.p2.y], color='b')

        if self.hole.contains_ball(self.gameball):
            hole = patches.Circle(self.hole.center.p, self.hole.R, edgecolor='g', facecolor='black')
            gameball = patches.Circle(self.gameball.center.p, self.gameball.R, edgecolor='none', facecolor='green')
        else:
            hole = patches.Circle(self.hole.center.p, self.hole.R, edgecolor='r', facecolor='black')
            gameball = patches.Circle(self.gameball.center.p, self.gameball.R, edgecolor='none', facecolor='red')
        ax.add_patch(hole)
        ax.add_patch(gameball)

    def move(self, direction: Point):
        if direction.len() < 1e-4:
            return

        options: List[Tuple[Point]] = list()
        for obstacle in self.obstacles:
            options.extend(obstacle.reflect_hit(self.gameball, direction))

        # TODO: make it a reusable function?
        options = sorted(options, key = lambda x: (x[0] - self.gameball.center).len())  # Choose the first hit to happen.
        assert options
        new_pos, new_direction = options[0]
        print(f'See: {new_pos} {new_direction}')

        self.gameball.center = new_pos
        self.track.append((self.gameball, self.current_step))
        self.move(new_direction)

    def reset(self):
        self.gameball = self.get_gameball()
        self.track = [self.gameball]
        self.current_step = 0
        info = {
            'track': self.track,
            'current_step': self.current_step,
        }

        return self.gameball, info

    def step(self, action):
        angle, power = action
        assert 0 <= angle < 360, f'angle must be between {0} and {360}'
        assert 0 < power <= 30, f'power must be between {0} and {30}'

        self.current_step += 1

        angle *= 2 * np.pi / 360
        direction = Point(np.cos(angle), np.sin(angle)) * power
        self.move(direction)

        observation = self.correct_gameball
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

