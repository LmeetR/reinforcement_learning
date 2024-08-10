import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100
HEIGHT = 5
WIDTH = 5


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Q Learning')
        # 创建窗口
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []

    def _build_canvas(self):
        """
        构建用于显示游戏界面的画布。

        该函数创建一个Canvas对象，并在其中绘制游戏所需的网格线和图形。
        """
        # 创建画布，设置背景颜色、高度和宽度
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)

        # 绘制网格线，垂直和水平方向上每隔UNIT距离绘制一条线
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400，步长为UNIT
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400，步长为UNIT
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # 在画布上添加图像，用于表示游戏中的不同形状
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        self.triangle1 = canvas.create_image(250, 150, image=self.shapes[1])
        self.triangle2 = canvas.create_image(150, 250, image=self.shapes[1])
        self.triangle3 = canvas.create_image(250, 350, image=self.shapes[1])
        self.circle = canvas.create_image(250, 250, image=self.shapes[2])

        # 将画布及其内容显示出来
        canvas.pack()

        # 返回创建的画布对象
        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("../img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(
            Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("../img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle

    def text_value(self, row, col, contents, action, font='Helvetica', size=10,
                   style='normal', anchor="nw"):
        if action == 0:
            origin_x, origin_y = 7, 42
        elif action == 1:
            origin_x, origin_y = 85, 42
        elif action == 2:
            origin_x, origin_y = 42, 5
        else:
            origin_x, origin_y = 42, 77

        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def print_value_all(self, q_table):
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for i in range(HEIGHT):
            for j in range(WIDTH):
                for action in range(0, 4):
                    state = [i, j]
                    if str(state) in q_table.keys():
                        temp = q_table[str(state)][action]
                        self.text_value(j, i, round(temp, 2), action)

    def coords_to_state(self, coords):
        """
        将坐标转换为状态编号。

        该方法接收一个坐标的列表，将坐标转换为状态的编号。
        坐标的转换通过将 x 和 y 轴上的值减去一个偏移量后，除以一个固定的比例来实现。
        这种转换使得我们可以将连续的坐标空间离散化，便于状态的处理和分析。

        参数:
        coords: 一个包含两个元素的列表，代表 x 和 y 坐标。

        返回值:
        一个包含两个整数的列表，代表转换后的状态编号。
        """
        # 计算 x 轴上的状态编号
        x = int((coords[0] - 50) / 100)
        # 计算 y 轴上的状态编号
        y = int((coords[1] - 50) / 100)
        # 返回转换后的状态编号
        # 就是把canvas的坐标，转化成物体的数组坐标，比如[50, 50] -> [0, 0]
        return [x, y]


    def state_to_coords(self, state):
        x = int(state[0] * 100 + 50)
        y = int(state[1] * 100 + 50)
        return [x, y]

    def reset(self):
        """
        重置环境到初始状态。

        此函数首先更新环境状态，然后暂停0.5秒，接着将矩形物体移动到画布的中心位置，
        最后渲染环境并返回当前状态。
        """
        # 更新环境状态
        self.update()
        # 暂停0.5秒，以便观察者可以看到移动过程
        time.sleep(0.5)
        # 获取矩形物体当前的坐标
        x, y = self.canvas.coords(self.rectangle)
        # 将矩形物体移动到画布的中心位置
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
        # 渲染环境，更新显示
        self.render()
        # 返回矩形物体当前的状态，用于下一步操作
        return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        """
        执行一个动作并返回结果状态、奖励和是否结束。

        参数:
        - action: int, 表示采取的动作，0-上，1-下，2-左，3-右。

        返回:
        - next_state: list, 下一个状态的坐标。
        - reward: int, 本次动作的得分。
        - done: bool, 是否游戏结束。
        """
        # 获取当前状态（矩形的位置坐标）
        state = self.canvas.coords(self.rectangle)
        # 初始化基础动作，不移动
        base_action = np.array([0, 0])
        # 当前结束的话，是否成功
        successful = 0
        # 渲染当前画面
        self.render()

        # 根据动作调整位置（绘图）
        if action == 0:  # 上
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 下
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 左
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # 右
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        # 执行移动
        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        # 将矩形提升到最上层，确保在其他图形之上
        self.canvas.tag_raise(self.rectangle)
        # 获取移动后的位置
        next_state = self.canvas.coords(self.rectangle)

        # 判断得分情况
        if next_state == self.canvas.coords(self.circle):
            # 到达目标，游戏结束，得分100
            reward = 100
            done = True
            successful = 1
        elif next_state in [self.canvas.coords(self.triangle1),
                            self.canvas.coords(self.triangle2),
                            self.canvas.coords(self.triangle3)]:
            # 碰撞到障碍物，游戏结束，得分-100
            reward = -100
            done = True
        else:
            # 未到达目标或碰撞，得分为0，游戏继续
            reward = 0
            done = False

        # 将坐标转换为状态
        next_state = self.coords_to_state(next_state)
        # 返回下一步状态、奖励和游戏是否结束
        return next_state, reward, done, successful

    # 渲染环境
    def render(self):
        time.sleep(0.03)
        self.update()
