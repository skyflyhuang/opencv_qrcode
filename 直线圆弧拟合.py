import numpy as np
import matplotlib.pyplot as plt

def line_func(x, m, c):
    """线性函数"""
    return m * x + c

def arc_func(r, x0, y0, theta_start, theta_end, num_points=100):
    """圆弧函数"""
    theta = np.linspace(theta_start, theta_end, num=num_points)
    x_arc = x0 + r * np.cos(theta)
    y_arc = y0 + r * np.sin(theta)
    return x_arc, y_arc

def connect_line_arc(line_start, line_end, arc_center, arc_radius, theta_start, theta_end):
    """连接直线和圆弧"""
    x_line = np.linspace(line_start[0], line_end[0], num=50)
    y_line = line_func(x_line, (line_end[1] - line_start[1]) / (line_end[0] - line_start[0]), line_start[1])

    x_arc, y_arc = arc_func(arc_radius, arc_center[0], arc_center[1], theta_start, theta_end)
    return np.concatenate((x_line, x_arc)), np.concatenate((y_line, y_arc))

# 定义直线和圆弧的参数
line_start = (1, 1)
line_end = (5, 4)
arc_center = (3, 0)
arc_radius = 2
theta_start = np.pi / 2
theta_end = np.pi

# 连接直线和圆弧
x_connect, y_connect = connect_line_arc(line_start, line_end, arc_center, arc_radius, theta_start, theta_end)

# 绘制连接结果
plt.figure(figsize=(8, 6))
plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'ro-', label='Line')
plt.plot(x_connect, y_connect, 'bo-', label='Connected Line-Arc')
plt.plot(arc_center[0], arc_center[1], 'go', label='Arc Center')
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Connecting Line and Arc')
plt.legend()
plt.grid(True)
plt.show()
