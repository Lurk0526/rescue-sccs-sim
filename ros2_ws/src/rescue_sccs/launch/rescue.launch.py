#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

# 5台机器人起始坐标（对应Python仿真的_robot_starts）
ROBOT_STARTS = [
    (20, 20), (20, 60), (60, 20), (60, 60), (40, 40)
]


def generate_launch_description():
    nodes = []

    # 1. 环境节点
    nodes.append(Node(
        package='rescue_sccs',
        executable='env_node',
        name='env_node',
        parameters=[{
            'width': 80, 'height': 80,
            'n_survivors': 50, 'n_clusters': 5,
            'life_decay_rate': 0.0018, 'seed': 42,
        }],
        output='screen',
    ))

    # 2. SCCS协调节点（延迟2秒等环境就绪）
    nodes.append(TimerAction(
        period=2.0,
        actions=[Node(
            package='rescue_sccs',
            executable='sccs_node',
            name='sccs_node',
            parameters=[{'n_robots': 5}],
            output='screen',
        )]
    ))

    # 3. 5台机器人节点（各自独立，延迟3秒）
    for i, (sx, sy) in enumerate(ROBOT_STARTS):
        nodes.append(TimerAction(
            period=3.0,
            actions=[Node(
                package='rescue_sccs',
                executable='robot_node',
                name=f'robot_node_{i}',
                parameters=[{
                    'robot_id': i,
                    'start_x': float(sx),
                    'start_y': float(sy),
                    'lidar_range': 8,
                    'detect_prob': 0.88,
                }],
                output='screen',
            )]
        ))

    # 4. Foxglove Bridge（可视化，延迟4秒）
    nodes.append(TimerAction(
        period=4.0,
        actions=[Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{'port': 8765}],
            output='screen',
        )]
    ))

    return LaunchDescription(nodes)
