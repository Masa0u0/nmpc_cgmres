import os
import os.path as osp
import numpy as np
from numpy import linalg as LA
from numpy.typing import NDArray
from argparse import ArgumentParser
from typing import Tuple, List
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from nmpc_cgmres import BaseEnv, BaseModel, BasePlanner, NmpcCgmres
from nmpc_cgmres.util import circle, runge_kutta_4


class DynamicPole:

    def __init__(self, x0: float, y0: float, vx: float, vy: float, radius: float) -> None:
        assert radius > 0.

        self.vx = vx
        self.vy = vy
        self.radius = radius

        self._x0 = x0
        self._y0 = y0

    def x(self, t: float) -> float:
        return self._x0 + self.vx * t

    def y(self, t: float) -> float:
        return self._y0 + self.vy * t


class DroneDynamicObstacleEnv(BaseEnv):

    X_DIM = 9
    U_DIM = 3
    MAX_ANGLE = np.deg2rad(30.)
    MAX_THRUST = 10.

    ALPHA_1X = 0.
    ALPHA_1Y = 0.
    ALPHA_1Z = 0.
    ALPHA_2X = 0.
    ALPHA_2Y = 0.
    ALPHA_2Z = 0.
    BETA_X = 1.
    BETA_Y = 1.
    BETA_Z = 1.

    def __init__(
        self,
        start_pos: NDArray,
        goal_pos: NDArray,
        poles: List[DynamicPole],
        dt: float,
        task_horizon: float,
    ):
        assert start_pos.shape == (3,)
        assert goal_pos.shape == (3,)
        assert dt > 0.
        assert task_horizon > 0

        super().__init__()

        self._start_pos = start_pos
        self._goal_pos = goal_pos
        self._poles = poles
        self._dt = dt
        self._max_step = int(task_horizon / dt)
        self._step_count = 0
        self._x = np.empty((self.X_DIM,))
        self._x_history = np.empty((self._max_step + 1, self.X_DIM))

        # 制御入力の制限
        self._u_lb = np.array([-self.MAX_ANGLE, -self.MAX_ANGLE, -self.MAX_THRUST])
        self._u_ub = np.array([self.MAX_ANGLE, self.MAX_ANGLE, self.MAX_THRUST])

        # ダイナミクス
        self._A = np.zeros((self.X_DIM, self.X_DIM))
        self._A[0:6, 3:9] = np.identity(6)
        self._A[6, 3] = self.ALPHA_1X
        self._A[7, 4] = self.ALPHA_1Y
        self._A[8, 5] = self.ALPHA_1Z
        self._A[6, 6] = self.ALPHA_2X
        self._A[7, 7] = self.ALPHA_2Y
        self._A[8, 8] = self.ALPHA_2Z
        self._B = np.zeros((self.X_DIM, self.U_DIM))
        self._B[6, 0] = self.BETA_X
        self._B[7, 1] = self.BETA_Y
        self._B[8, 2] = self.BETA_Z

    def reset(self) -> NDArray:
        self._step_count = 0
        self._x = np.r_[self._start_pos, np.zeros((6,))]
        self._x_history = np.empty((self._max_step + 1, self.X_DIM))
        self._x_history[0, :] = self._x[:]

        return self._x.copy()

    def step(self, u: NDArray) -> Tuple[NDArray, float, bool, dict]:
        assert u.shape == (self.U_DIM,)

        # dynamics
        u = np.clip(u, self._u_lb, self._u_ub)
        self._x = runge_kutta_4(self._x, u, self._x_dot, self._dt)

        # cost
        cost = None  # 特に使わないためグローバルコストは定めない

        # done
        self._step_count += 1
        done = self._step_count >= self._max_step

        # info: 現在の時刻を0としたときの障害物の位置と速度
        t = self._dt * self._step_count
        info = {'poles': []}
        for i in range(len(self._poles)):
            x0 = self._poles[i].x(t)
            y0 = self._poles[i].y(t)
            vx = self._poles[i].vx
            vy = self._poles[i].vy
            radius = self._poles[i].radius
            info['poles'].append(DynamicPole(x0, y0, vx, vy, radius))

        # 記録
        self._x_history[self._step_count, :] = self._x[:]

        return self._x.copy(), cost, done, info

    def _x_dot(self, x: NDArray, u: NDArray) -> NDArray:
        return self._A @ x + self._B @ u

    def _set_imgs(self, axis: Axes):
        self._imgs.clear()
        self._imgs['drone'] = axis.plot([], [], marker='o', c='r', markersize=10)[0]
        self._imgs['traj'] = axis.plot([], [], c='k', linestyle='dashed')[0]
        self._imgs['start'] = axis.plot([], [], marker='o', c='k', markersize=10)[0]
        self._imgs['goal'] = axis.plot([], [], marker='o', c='k', markersize=10)[0]

        for i in range(len(self._poles)):
            self._imgs[f'pole{i}'] = axis.plot([], [], marker='o', c='k', markersize=10)[0]
            self._imgs[f'barrier{i}'] = axis.plot([], [], c='k', linestyle='dashed')[0]

        margin = 1.
        x_min = min(self._start_pos[0], self._goal_pos[0])
        x_max = max(self._start_pos[0], self._goal_pos[0])
        y_min = min(self._start_pos[1], self._goal_pos[1])
        y_max = max(self._start_pos[1], self._goal_pos[1])
        axis.set_xlim([x_min - margin, x_max + margin])
        axis.set_ylim([y_min - margin, y_max + margin])

    def _plot_func(self, i: int):
        self._imgs['drone'].set_data(self._x_history[i, 0], self._x_history[i, 1])
        self._imgs['traj'].set_data(self._x_history[:i, 0], self._x_history[:i, 1])
        self._imgs['start'].set_data(self._start_pos[0], self._start_pos[1])
        self._imgs['goal'].set_data(self._goal_pos[0], self._goal_pos[1])

        t = self._dt * i
        for i in range(len(self._poles)):
            self._imgs[f'pole{i}'].set_data(self._poles[i].x(t), self._poles[i].y(t))
            self._imgs[f'barrier{i}'].set_data(
                *circle(self._poles[i].x(t), self._poles[i].y(t), self._poles[i].radius)
            )

    @property
    def max_step(self) -> int:
        return self._max_step

    @property
    def n_frames(self) -> int:
        return len(self._x_history)


class DroneDynamicObstacleModel(BaseModel):

    X_DIM = 9
    U_DIM = 3
    C_DIM = 0

    MAX_ANGLE = np.deg2rad(30.)
    MAX_THRUST = 10.

    ALPHA_1X = 0.
    ALPHA_1Y = 0.
    ALPHA_1Z = 0.
    ALPHA_2X = 0.
    ALPHA_2Y = 0.
    ALPHA_2Z = 0.
    BETA_X = 1.
    BETA_Y = 1.
    BETA_Z = 1.

    def __init__(
        self,
        vel_weights: NDArray,
        input_weight: float,
        rx: float,
        ru: float,
    ) -> None:
        assert vel_weights.shape == (3,) and np.all(vel_weights > 0.)
        assert input_weight > 0.  # コスト関数の凸性を保証するために正則化項は必須
        assert rx > 0.
        assert ru > 0.

        super().__init__()

        self._rx = rx
        self._ru = ru

        # ダイナミクス(この例ではEnvと全く同じ正確なダイナミクスが利用可能とする)
        self._A = np.zeros((self.X_DIM, self.X_DIM))
        self._A[0:6, 3:9] = np.identity(6)
        self._A[6, 3] = self.ALPHA_1X
        self._A[7, 4] = self.ALPHA_1Y
        self._A[8, 5] = self.ALPHA_1Z
        self._A[6, 6] = self.ALPHA_2X
        self._A[7, 7] = self.ALPHA_2Y
        self._A[8, 8] = self.ALPHA_2Z
        self._B = np.zeros((self.X_DIM, self.U_DIM))
        self._B[6, 0] = self.BETA_X
        self._B[7, 1] = self.BETA_Y
        self._B[8, 2] = self.BETA_Z

        # 速度にのみ重みをかける
        self._Q = np.zeros((self.X_DIM, self.X_DIM))
        self._Q[3, 3] = vel_weights[0]
        self._Q[4, 4] = vel_weights[1]
        self._Q[5, 5] = vel_weights[2]
        self._R = np.diag([input_weight] * self.U_DIM)

        self._poles = []

    def x_dot(self, x: NDArray, u: NDArray):
        assert x.shape == (self.X_DIM,)
        assert u.shape == (self.U_DIM,)

        xd = self._A @ x + self._B @ u
        return xd

    def gradient_terminal_cost_state(self, t: float, x: NDArray, x_des: NDArray) -> NDArray:
        assert t >= 0.
        assert x.shape == (self.X_DIM,)
        assert x_des.shape == (self.X_DIM,)

        # この例では特に終端コストを考えていないため，gradient_hamiltonian_stateの結果とほぼ同じになる
        pBpx_sum = np.zeros((self.X_DIM,))
        for pole in self._poles:
            pole_x = pole.x(t)
            pole_y = pole.y(t)
            denom = (pole.radius**2 - ((x[0] - pole_x)**2 + (x[1] - pole_y)**2))**3
            pBpx_sum[0] += 4. * (x[0] - pole_x) / denom
            pBpx_sum[1] += 4. * (x[1] - pole_y) / denom

        res = self._Q @ (x - x_des) + (1. / self._rx) * pBpx_sum
        return res

    def gradient_hamiltonian_state(
        self,
        t: float,
        x: NDArray,
        x_des: NDArray,
        u: NDArray,
        lam: NDArray,
        rho: NDArray,
    ) -> NDArray:
        assert t >= 0.
        assert x.shape == (self.X_DIM,), x.shape
        assert x_des.shape == (self.X_DIM,), x_des.shape
        assert lam.shape == (self.X_DIM,), lam.shape

        pBpx_sum = np.zeros((self.X_DIM,))
        for pole in self._poles:
            pole_x = pole.x(t)
            pole_y = pole.y(t)
            denom = (pole.radius**2 - ((x[0] - pole_x)**2 + (x[1] - pole_y)**2))**3
            pBpx_sum[0] += 4. * (x[0] - pole_x) / denom
            pBpx_sum[1] += 4. * (x[1] - pole_y) / denom

        pHpx = self._Q @ (x - x_des) + (1. / self._rx) * pBpx_sum + self._A.T @ lam
        return pHpx

    def gradient_hamiltonian_input(
        self,
        t: float,
        x: NDArray,
        x_des: NDArray,
        u: NDArray,
        lam: NDArray,
        rho: NDArray,
    ) -> NDArray:
        assert u.shape == (self.U_DIM,), u.shape
        assert lam.shape == (self.X_DIM,), lam.shape

        pPpu = (4. * max(0., u[0]**2 + u[1]**2 - self.MAX_ANGLE**2)) * np.array([u[0], u[1], 0.])

        pHpu = self._R @ u + self._ru * pPpu + self._B.T @ lam
        return pHpu

    def constraint(self, t: float, x: NDArray, x_des: NDArray, u: NDArray) -> NDArray:
        return np.zeros((self.C_DIM,))

    def update_poles(self, poles: List[DynamicPole]) -> None:
        self._poles = poles


class DroneDynamicObstaclePlanner(BasePlanner):

    X_DIM = 9
    U_DIM = 3
    EPS = 1e-9

    def __init__(self, pred_len: int, max_speed: float, dist_threshold: float):
        assert max_speed > 0.
        assert dist_threshold > 0.

        super().__init__(pred_len=pred_len)

        self._max_speed = max_speed
        self._dist_threshold = dist_threshold

    def plan(self, x: NDArray, x_goal: NDArray) -> NDArray:
        assert x.shape == (self.X_DIM,), x.shape
        assert x_goal.shape == (self.X_DIM,), x_goal.shape

        pos_cur = x[0:3]
        pos_goal = x_goal[0:3]

        dist = LA.norm(pos_cur - pos_goal)
        if dist > self._dist_threshold:
            speed = self._max_speed
        else:
            ratio = dist / self._dist_threshold
            speed = ratio * (2. - ratio) * self._max_speed

        # 1ステップごとの目標状態
        vel_des = (pos_goal - pos_cur) * (speed / (dist + self.EPS))
        x_des = np.zeros((self.X_DIM,))
        x_des[3:6] = vel_des

        # 予測区間全体の目標状態．予測区間に渡って一定とする．
        X_des = np.tile(x_des, (self._pred_len, 1))

        return X_des


def run(
    env: DroneDynamicObstacleEnv,
    model: DroneDynamicObstacleModel,
    planner: DroneDynamicObstaclePlanner,
    controller: NmpcCgmres,
    x_goal: NDArray,
    dt: float,
) -> Tuple[NDArray, NDArray]:
    done = False
    x = env.reset()
    poles = []
    history_x, history_u = [], []
    step_count = 0

    while not done:
        # 別モジュールで検出した障害物の位置と速度をモデルに教える
        model.update_poles(poles)

        # 局所計画
        x_des = planner.plan(x, x_goal)

        # 最適制御入力を計算
        u = controller.step(x, x_des, dt)

        # プラントを進める
        next_x, cost, done, info = env.step(u)

        # 記録
        history_x.append(x.copy())
        history_u.append(u.copy())

        # 状態の更新
        x = next_x
        poles = info['poles']  # 別モジュールで柱の位置と速度を検出する想定
        step_count += 1

        # 経過表示
        print('\r' + f'Step: {step_count}/{env.max_step}', end='')

    print()

    return np.array(history_x), np.array(history_u)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--start_pos', type=float, nargs=3, default=[-2., 0., 0.])
    parser.add_argument('--goal_pos', type=float, nargs=3, default=[2., 0., 1.])
    parser.add_argument('--pole_init_x', type=float, nargs='*', default=[2.])
    parser.add_argument('--pole_init_y', type=float, nargs='*', default=[0.5])
    parser.add_argument('--pole_vx', type=float, nargs='*', default=[-0.4])
    parser.add_argument('--pole_vy', type=float, nargs='*', default=[-0.1])
    parser.add_argument('--pole_radius', type=float, nargs='*', default=[0.4])
    parser.add_argument('--vel_weights', type=float, nargs=3, default=[1., 1., 1.])
    parser.add_argument('--input_weight', type=float, default=1e-2)
    parser.add_argument('--rx', type=float, default=100.)
    parser.add_argument('--ru', type=float, default=1.)
    parser.add_argument('--controller_freq', type=float, default=100.)
    parser.add_argument('--task_horizon', type=float, default=20.)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--max_speed', type=float, default=0.5)
    parser.add_argument('--dist_threshold', type=float, default=1.)
    parser.add_argument('--pred_horizon', type=float, default=2.)
    parser.add_argument('--horizon_half_time', type=float, default=0.3)
    parser.add_argument('--zeta_coef', type=float, default=1.)
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--result_dir', default=osp.join(osp.dirname(__file__), '../result'))
    args = parser.parse_args()

    n_pole = len(args.pole_init_x)
    assert len(args.pole_init_x) == n_pole
    assert len(args.pole_init_y) == n_pole
    assert len(args.pole_vx) == n_pole
    assert len(args.pole_vy) == n_pole
    assert len(args.pole_radius) == n_pole

    dt = 1. / args.controller_freq  # 制御周期

    # 障害物のリストを作成
    poles = []
    for i in range(n_pole):
        poles.append(DynamicPole(
            x0=args.pole_init_x[i],
            y0=args.pole_init_y[i],
            vx=args.pole_vx[i],
            vy=args.pole_vy[i],
            radius=args.pole_radius[i],
        ))

    env = DroneDynamicObstacleEnv(
        start_pos=np.array(args.start_pos),
        goal_pos=np.array(args.goal_pos),
        poles=poles,
        dt=dt,
        task_horizon=args.task_horizon,
    )
    model = DroneDynamicObstacleModel(
        vel_weights=np.array(args.vel_weights),
        input_weight=args.input_weight,
        rx=args.rx,
        ru=args.ru,
    )
    planner = DroneDynamicObstaclePlanner(
        pred_len=args.pred_len,
        max_speed=args.max_speed,
        dist_threshold=args.dist_threshold,
    )
    controller = NmpcCgmres(
        model=model,
        pred_horizon=args.pred_horizon,
        pred_len=args.pred_len,
        horizon_half_time=args.horizon_half_time,
        zeta_coef=args.zeta_coef,
    )

    # シミュレーション
    x_goal = np.array(args.goal_pos + [0.] * 6)
    history_x, history_u = run(env, model, planner, controller, x_goal, dt)

    # 結果を保存
    if args.save_result:
        os.makedirs(args.result_dir, exist_ok=True)
        header = datetime.now().strftime("%Y-%m-%d-%H%M%S")

        # Save plot
        t = np.linspace(0., args.task_horizon, env.n_frames - 1)

        plt.figure()
        plt.plot(t, history_x[:, 0], label='x')
        plt.plot(t, history_x[:, 1], label='y')
        plt.plot(t, history_x[:, 2], label='z')
        plt.xlabel('Time [sec]')
        plt.ylabel('Position [m]')
        plt.legend()
        plt.savefig(osp.join(args.result_dir, f'{header}_state.png'))

        plt.figure()
        plt.plot(t, history_u[:, 0], label='u0')
        plt.plot(t, history_u[:, 1], label='u1')
        plt.plot(t, history_u[:, 2], label='u2')
        plt.xlabel('Time [sec]')
        plt.ylabel('System Input')
        plt.legend()
        plt.savefig(osp.join(args.result_dir, f'{header}_input.png'))

        # Save animation
        env.save_animation(
            save_path=osp.join(args.result_dir, f'{header}_trajectory.mp4'),
            dt=dt,
            n_frames=env.n_frames,
        )
