from numpy.typing import NDArray
from abc import ABC, abstractmethod


class BasePlanner(ABC):
    """ 予測区間の目標状態を計画するモジュールの基底クラス． """

    def __init__(self, pred_len: int):
        """
        BasePlannerのコンストラクタ．

        Parameters
        ----------
        pred_len : int
            予測ステップ数
        """
        assert pred_len > 0
        self._pred_len = pred_len

    @abstractmethod
    def plan(self, x: NDArray, x_goal: NDArray):
        """
        現在の状態と長期的な目標状態から，予測区間内の目標状態を作成する．

        Parameters
        ----------
        x : NDArray[shape=(state_size,)]
            現在の状態
        x_goal : NDArray[shape=(state_size,)]
            長期的な目標状態

        Returns
        -------
        x_des : NDArray[shape=(pred_len, state_size)]
            予測区間内の目標状態
        """
        raise NotImplementedError()
