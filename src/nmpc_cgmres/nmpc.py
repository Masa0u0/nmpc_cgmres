import numpy as np
from numpy.typing import NDArray

from .model import BaseModel
from .util import runge_kutta_4, cgmres


class NmpcCgmres:
    """
    非線形モデル予測制御．
    非線形最適制御入門のアルゴリズム8.3(https://www.coronasha.co.jp/np/isbn/9784339033182/)
    """

    def __init__(
        self,
        model: BaseModel,
        pred_horizon: float,
        pred_len: int,
        horizon_half_time: float,
        threshold: float = 1e-6,
        delta: float = 1e-6,
        zeta_coef: float = 1.,
        init_u: NDArray = None,
    ):
        """
        NmpcCgmresのコンストラクタ．

        Parameters
        ----------
        model : BaseModel
            制御対象の構造を記述したモデル
        pred_horizon : float
            予測区間[sec]
        pred_len : int
            予測区間の離散化分割数
        horizon_half_time : float
            初期に予測区間を0から伸ばしていく際の，pred_horizonの半分に到達するまでの時間[sec]
        threshold : float, default 1e-3
            C/GMRESの収束判定の閾値
        delta : float, default 1e-6
            数値微分に用いる微少量
        zeta_coef : float, default 0.1
            p.179でzeta = zeta_coef / dtとする．zeta_coef = 1で収束が保証されているらしい．
        init_u : NDArray, default None
            制御入力の初期値
        """
        assert pred_horizon > 0., pred_horizon
        assert pred_len > 0, pred_len
        assert horizon_half_time > 0., horizon_half_time
        assert threshold > 0., threshold
        assert delta > 0., delta
        assert zeta_coef > 0., zeta_coef
        assert init_u is None or init_u.shape == (self._u_dim,), init_u.shape

        self._model = model
        self._x_dim = self._model.X_DIM
        self._u_dim = self._model.U_DIM
        self._c_dim = self._model.C_DIM
        self._pred_horizon = pred_horizon
        self._pred_len = pred_len
        self._alpha = np.log(2.) / horizon_half_time
        self._threshold = threshold
        self._zeta_coef = zeta_coef
        self._delta = delta
        self._var_dim = self._pred_len * (self._u_dim + self._c_dim)  # (8.42)の決定変数の次元

        self._dt = 0.  # 離散化間隔
        self._time = 0.
        if init_u is None:
            self._U = np.zeros((self._pred_len, self._u_dim))
        else:
            self._U = np.tile(init_u, (pred_len, 1))

        # Rhoの初期値を0.01よりも大きな値にすると数値エラーを防げるらしい # TODO: なぜ？
        self._Rho = np.full((self._pred_len, self._c_dim), 0.011)
        self._var = np.zeros((self._var_dim,))  # (8.42)の決定変数(U, Rho)

    def step(
        self,
        x_cur: NDArray,
        X_des: NDArray,
        time_from_last_call: float,
    ) -> NDArray:
        """
        p.176のアルゴリズム8.3により最適制御入力を求める．

        Parameters
        ----------
        x_cur : NDArray[shape=(x_dim,)]
            現在の状態
        X_des : NDArray[shape=(pred_len, x_dim)]
            目標状態の軌道
        time_from_last_call : float
            前回の呼び出しからの経過時間[sec]

        Returns
        -------
        u_opt : NDArray[shape=(u_dim,)]
            最適制御入力
        """
        assert x_cur.shape == (self._x_dim,), x_cur.shape
        assert X_des.shape == (self._pred_len, self._x_dim), X_des.shape
        assert time_from_last_call > 0., time_from_last_call

        # 時間経過とともに予測区間を伸ばしていく(p.166)
        # それに伴い離散化間隔も徐々に長くなる
        self._time += time_from_last_call
        self._dt = self._pred_horizon * (1. - np.exp(-self._alpha * self._time)) / self._pred_len

        # (8.42)の右辺を計算．数値微分はp.178を参照．
        dx = self._model.x_dot(x_cur, self._U[0]) * self._delta
        Fxt = self._calc_F(x_cur + dx, X_des, self._U, self._Rho).flatten()
        F = self._calc_F(x_cur, X_des, self._U, self._Rho).flatten()
        zeta = self._zeta_coef / time_from_last_call
        right = -zeta * F - ((Fxt - F) / self._delta)

        # C/GMRESによりUとRhoの時間変化率を更新
        # FIXED: 教科書通り前回の解を初期推定解をするように修正
        def calc_left(var: NDArray) -> NDArray:
            """ 決定変数(U, Rho)から(8.42)の左辺を計算する """
            assert var.shape == (self._var_dim,)

            var_reshaped = var.reshape((self._pred_len, -1))
            dU = var_reshaped[:, :self._u_dim] * self._delta
            dRho = var_reshaped[:, self._u_dim:] * self._delta

            Fuxt = self._calc_F(x_cur + dx, X_des, self._U + dU, self._Rho + dRho).flatten()
            left = (Fuxt - Fxt) / self._delta
            return left
        self._var = cgmres(calc_left, right, self._var, self._threshold)
        var = self._var.reshape((self._pred_len, -1))
        U_dot = var[:, :self._u_dim]
        Rho_dot = var[:, self._u_dim:]

        # 追跡変数を更新
        # FIXED: 元実装は数値微分に用いる微少量であるdeltaで更新していたが．それは誤り．
        self._U += U_dot * time_from_last_call
        self._Rho += Rho_dot * time_from_last_call

        u_opt = self._U[0]
        return u_opt

    def _calc_F(
        self,
        x_cur: NDArray,
        X_des: NDArray,
        U: NDArray,
        Rho: NDArray,
    ) -> NDArray:
        """ (8.39)のFを計算する． """
        assert x_cur.shape == (self._x_dim,), x_cur.shape
        assert X_des.shape == (self._pred_len, self._x_dim), X_des.shape
        assert U.shape == (self._pred_len, self._u_dim), U.shape
        assert Rho.shape == (self._pred_len, self._c_dim), Rho.shape

        X_pred = self._predict_x_traj(x_cur, U)  # (pred_len + 1, x_dim)
        Lam_pred = self._predict_lam_traj(X_pred, X_des, U, Rho)  # (pred_len, x_dim)
        F = self._gradient_hamiltonian_input_with_constraint(X_pred, X_des, U, Lam_pred, Rho)
        return F

    def _predict_x_traj(self, x_cur: NDArray, U: NDArray) -> NDArray:
        """ 現在の状態と制御入力の軌道から状態の軌道を予測する． """
        assert x_cur.shape == (self._x_dim,), x_cur.shape
        assert U.shape == (self._pred_len, self._u_dim), U.shape

        X_pred = np.empty((self._pred_len + 1, self._x_dim))
        X_pred[0, :] = x_cur[:]

        # FIXED: 元実装はpred_len回np.concatenateを呼んでおり，計算量が無駄に多くなっていた
        for k in range(0, self._pred_len):
            X_pred[k + 1, :] = runge_kutta_4(X_pred[k, :], U[k, :], self._model.x_dot, self._dt)

        return X_pred

    def _predict_lam_traj(
        self,
        X_pred: NDArray,
        X_des: NDArray,
        U: NDArray,
        Rho: NDArray,
    ) -> NDArray:
        """ 状態随伴変数の軌道を予測する． """
        assert X_pred.shape == (self._pred_len + 1, self._x_dim), X_pred.shape
        assert X_des.shape == (self._pred_len, self._x_dim), X_des.shape
        assert U.shape == (self._pred_len, self._u_dim), U.shape
        assert Rho.shape == (self._pred_len, self._c_dim), Rho.shape

        # pred final adjoint state
        Lam_pred = np.empty((self._pred_len, self._x_dim))
        Lam_pred[-1, :] = self._predict_terminal_lam(X_pred[-1], X_des[-1], Rho[-1])

        # FIXED: 元実装はpred_len回np.concatenateを呼んでおり，計算量が無駄に多くなっていた
        for k in range(self._pred_len - 1, 0, -1):
            Lam_pred[k - 1, :] = self._predict_prev_lam(
                k, X_pred[k], X_des[k], U[k], Lam_pred[k], Rho[k]
            )

        return Lam_pred

    def _predict_prev_lam(
        self,
        k: int,
        x: NDArray,
        x_des: NDArray,
        u: NDArray,
        lam: NDArray,
        rho: NDArray,
    ) -> NDArray:
        """ 1ステップ前の状態随伴変数を予測する． """
        assert 1 <= k <= self._pred_len - 1, k
        assert x.shape == (self._x_dim,), x.shape
        assert x_des.shape == (self._x_dim,), x_des.shape
        assert u.shape == (self._u_dim,), u.shape
        assert lam.shape == (self._x_dim,), lam.shape
        assert rho.shape == (self._c_dim,), rho.shape

        t = self._dt * k
        delta_lam = self._dt * self._model.gradient_hamiltonian_state(t, x, x_des, u, lam, rho)
        prev_lam = lam + delta_lam
        return prev_lam

    def _predict_terminal_lam(
        self,
        terminal_x: NDArray,
        terminal_x_des: NDArray,
        rho: NDArray,
    ) -> NDArray:
        """ 状態随伴変数の終端値を予測する． """
        assert terminal_x.shape == (self._x_dim,), terminal_x.shape
        assert terminal_x_des.shape == (self._x_dim,), terminal_x_des.shape
        assert rho.shape == (self._c_dim,), rho.shape

        t = self._dt * self._pred_len
        terminal_lam = self._model.gradient_terminal_cost_state(t, terminal_x, terminal_x_des)
        return terminal_lam

    def _gradient_hamiltonian_input_with_constraint(
        self,
        X_pred: NDArray,
        X_des: NDArray,
        U: NDArray,
        Lam: NDArray,
        Rho: NDArray,
    ) -> NDArray:
        """ (8.39)のF，すなわちHの(U, Rho)による偏微分を求める． """
        assert X_pred.shape == (self._pred_len + 1, self._x_dim), X_pred.shape
        assert X_des.shape == (self._pred_len, self._x_dim), X_des.shape
        assert U.shape == (self._pred_len, self._u_dim), U.shape
        assert Lam.shape == (self._pred_len, self._x_dim), Lam.shape
        assert Rho.shape == (self._pred_len, self._c_dim), Rho.shape

        F_u = np.zeros((self._pred_len, self._u_dim))
        F_c = np.zeros((self._pred_len, self._c_dim))

        for k in range(0, self._pred_len):
            t = self._dt * k
            F_u[k, :] = self._model.gradient_hamiltonian_input(
                t, X_pred[k + 1], X_des[k], U[k], Lam[k], Rho[k]
            )
            F_c[k, :] = self._model.constraint(t, X_pred[k + 1], X_des[k], U[k])

        F = np.concatenate([F_u, F_c], axis=1)
        return F
