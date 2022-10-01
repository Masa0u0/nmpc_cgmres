from numpy.typing import NDArray
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """ 制御モデルの構造を規定するモジュールの基底クラス． """

    X_DIM = None  # 状態の次元
    U_DIM = None  # 制御入力の次元
    C_DIM = None  # 等式制約の次元

    def __init__(self) -> None:
        """ BaseModelのコンストラクタ． """
        assert self.X_DIM is not None and self.X_DIM > 0
        assert self.U_DIM is not None and self.U_DIM > 0
        assert self.C_DIM is not None and 0 <= self.C_DIM <= self.U_DIM

    @abstractmethod
    def x_dot(self, x: NDArray, u: NDArray) -> NDArray:
        """
        連続時間の状態方程式．

        Parameters
        ----------
        x : NDArray[shape=(x_dim,)]
            現在の状態
        u : NDArray[shap=(u_dim,)]
            現在の制御入力

        Returns
        -------
        xd : NDArray[shape=(x_dim,)]
            状態の時間変化率
        """
        raise NotImplementedError()

    @abstractmethod
    def gradient_terminal_cost_state(t: float, self, x: NDArray, x_des: NDArray) -> NDArray:
        """
        終端コストの状態による偏微分を計算する．

        Parameters
        ----------
        t : float
            現在時刻を0としたときの時刻
        x : NDArray[shape=(x_dim,)]
            状態
        x_des : NDArray[shape=(x_dim,)]
            目標状態

        Returns
        -------
        terminal_lam : NDArray[shape=(x_dim,)]
            終端コストの状態による偏微分
        """
        raise NotImplementedError()

    @abstractmethod
    def gradient_hamiltonian_state(
        self,
        t: float,
        x: NDArray,
        x_des: NDArray,
        u: NDArray,
        lam: NDArray,
        rho: NDArray,
    ) -> NDArray:
        """
        ハミルトン関数の状態による偏微分を計算する．

        Parameters
        ----------
        t : float
            現在時刻を0としたときの時刻
        x : NDArray[shape=(x_dim,)]
            状態
        x_des : NDArray[shape=(x_dim,)]
            目標状態
        u : NDArray[shape=(u_dim,)]
            制御入力
        lam : NDArray[shape=(x_dim,)]
            状態方程式の随伴変数
        rho : NDArray[shape=(c_dim,)]
            等式制約の随伴変数

        Returns
        -------
        pH/px : NDArray[shape=(x_dim,)]
            ハミルトン関数の状態による偏微分
        """
        raise NotImplementedError()

    @abstractmethod
    def gradient_hamiltonian_input(
        self,
        t: float,
        x: NDArray,
        x_des: NDArray,
        u: NDArray,
        lam: NDArray,
        rho: NDArray,
    ) -> NDArray:
        """
        ハミルトン関数の制御入力による偏微分を計算する．

        Parameters
        ----------
        t : float
            現在時刻を0としたときの時刻
        x : NDArray[shape=(x_dim,)]
            状態
        x_des : NDArray[shape=(x_dim,)]
            目標状態
        u : NDArray[shape=(u_dim,)]
            制御入力
        lam : NDArray[shape=(x_dim,)]
            状態方程式の随伴変数
        rho : NDArray[shape=(c_dim,)]
            等式制約の随伴変数

        Returns
        -------
        pH/pu : NDArray[shape=(u_dim,)]
            ハミルトン関数の制御入力による偏微分
        """
        raise NotImplementedError()

    @abstractmethod
    def constraint(self, t: float, x: NDArray, x_des: NDArray, u: NDArray) -> NDArray:
        """
        常に0と等しくなる等式制約を計算する．

        Parameters
        ----------
        t : float
            現在時刻を0としたときの時刻
        x : NDArray[shape=(x_dim,)]
            状態
        x_des : NDArray[shape=(x_dim,)]
            目標状態
        u : NDArray[shape=(u_dim,)]
            制御入力

        Returns
        -------
        constraint : NDArray[shape=(c_dim,)]
            常に0と等しくなる等式制約
        """
        raise NotImplementedError()
