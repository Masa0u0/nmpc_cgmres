import numpy as np
from numpy import linalg as LA
from numpy.typing import NDArray
from typing import Tuple, Callable
from rich import print as rprint


def circle(
    center_x: float,
    center_y: float,
    radius: float,
    start: float = 0.,
    end: float = 2. * np.pi,
    n_point: int = 100,
) -> Tuple[NDArray, NDArray]:
    """
    円を構成する点群を作成する．

    Parameters
    ----------
    center_x : float
        中心のx座標
    center_y : float
        中心のy座標
    radius : float
        円の半径
    start : float, default 0.
        開始角度[rad]
    end : float, default 2. * np.pi
        終了角度[rad]
    n_point : int, default 100.
        点の個数

    Returns
    -------
    circle xs : NDArray[shape=(n_point,)]
        点群のx座標
    circle ys : NDArray[shape=(n_point,)]
        点群のy座標

    """
    assert radius > 0., radius
    assert 0. <= start < end <= 2. * np.pi, (start, end)

    diff = end - start

    circle_xs = []
    circle_ys = []

    for i in range(n_point + 1):
        circle_xs.append(center_x + radius * np.cos(i * diff / n_point + start))
        circle_ys.append(center_y + radius * np.sin(i * diff / n_point + start))

    return np.array(circle_xs), np.array(circle_ys)


def runge_kutta_4(
    x: NDArray,
    u: NDArray,
    f: Callable[[NDArray, NDArray], NDArray],
    dt: float,
) -> NDArray:
    """
    4次のルンゲクッタ法により，現在の状態と制御入力から次の状態を計算する．

    Parameters
    ----------
    x : NDArray[shape=(x_dim,)]
        現在の状態
    u : NDArray[shap=(u_dim,)]
        現在の制御入力
    f : Callable[[NDArray, NDArray], NDArray]
        連続時間状態方程式．xd = f(x, u)を満たす．
    dt : float
        積分時間

    Returns
    -------
    x_next : NDArray[shape=(x_dim,)]
        dt後の状態
    """
    assert x.ndim == 1, x.shape
    assert u.ndim == 1, u.shape
    assert dt > 0., dt

    k1 = f(x, u) * dt
    k2 = f(x + k1 / 2., u) * dt
    k3 = f(x + k2 / 2., u) * dt
    k4 = f(x + k3, u) * dt

    x_next = x + (k1 + 2. * k2 + 2. * k3 + k4) / 6.
    return x_next


def cgmres(
    calc_left: Callable[[NDArray], NDArray],
    right: NDArray,
    x0: NDArray,
    threshold: float,
) -> NDArray:
    """
    C/GMRES法により行列方程式(Ax = b)を解く．

    Parameters
    ----------
    calc_left : Callable[[NDArray[shape=(x_dim,)]], NDArray[shape=(eq_dim,)]]
        決定変数xから行列方程式の左辺を計算する関数
    right : NDArray[shape=(eq_dim,)]
        行列方程式の右辺
    x0 : NDArray[shape=(x_dim,)]
        初期推定解
    threshold : float
        収束判定の閾値

    Returns
    -------
    x : NDArray[shape=(x_dim,)]
        行列方程式の解
    """
    assert right.ndim == 1, right.shape
    assert x0.ndim == 1, x0.shape
    assert threshold > 0., threshold

    var_dim = x0.shape[0]

    r0 = right - calc_left(x0)  # 初期残渣
    r0_norm = LA.norm(r0)

    vs = np.zeros((var_dim, var_dim + 1))
    vs[:, 0] = r0 / r0_norm
    hs = np.zeros((var_dim + 1, var_dim + 1))
    e = np.zeros((var_dim + 1,))
    e[0] = 1.

    for i in range(0, var_dim):
        Av = calc_left(vs[:, i])
        sum_Av = np.zeros((var_dim,))

        for j in range(0, i + 1):
            hs[j, i] = Av @ vs[:, j]
            sum_Av += hs[j, i] * vs[:, j]

        v_est = Av - sum_Av
        hs[i + 1, i] = LA.norm(v_est)
        vs[:, i + 1] = v_est / hs[i + 1, i]
        ys = LA.lstsq(hs[:i + 1, :i], r0_norm * e[:i + 1], rcond=None)[0]

        error = LA.norm(r0_norm * e[:i + 1] - hs[:i + 1, :i] @ ys[:i])  # ||Ax - b||
        if error < threshold:
            return x0 if i == 0 else x0 + vs[:, :i - 1] @ ys_pre[:i - 1]

        ys_pre = ys
    else:
        # FIXME: 収束しなかった場合に誤った解を返すのはまずい．時間がかかっても別の手法で解くようにする．
        rprint(f'[red]Error: C/GMRES failed to converge. ||Ax - b|| = {error:.2e}[/red]')
        return x0 + vs[:, :var_dim - 1] @ ys_pre[:var_dim - 1]
