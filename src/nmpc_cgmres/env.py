from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Tuple, final
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes


class BaseEnv(ABC):
    """
    シミュレーション用の環境の基底クラス． \\
    Gazeboなど他のシミュレータがある場合は使用しなくてよい． \\
    本来はブラックボックスであり，コントローラ側から内部を参照することはできない．
    """

    def __init__(self) -> None:
        """ BaseEnvのコンストラクタ． """
        self._imgs = dict()

    @abstractmethod
    def reset(self) -> NDArray:
        """
        環境を初期化する．

        Returns
        -------
        init_x : NDArray[shape=(x_dim,)]
            初期状態
        """
        self._imgs.clear()

    @abstractmethod
    def step(self, u: NDArray) -> Tuple[NDArray, float, bool, dict]:
        """
        環境を1ステップ進める．

        Parameters
        ----------
        u : NDArray[shape=(u_dim,)]
            制御入力

        Returns
        -------
        next_x : NDArray[shape=(x_dim,)]
            次の状態
        cost : float
            現在のステップにおけるコスト
        done : bool
            終了フラグ
        info : dict
            その他情報
        """
        raise NotImplementedError()

    def _set_imgs(self, axis: Axes) -> None:
        """
        self._imgsの初期設定を行う．

        Parameters
        ----------
        axis : Axes
            プロット用のグラフ
        """
        raise NotImplementedError()

    def _plot_func(self, i: int) -> None:
        """
        self._imgsにiフレーム目の画像をプロットする．

        Parameters
        ----------
        i : int
            フレーム番号
        """
        raise NotImplementedError()

    @final
    def get_animation(self, dt: float, n_frames: int) -> animation.FuncAnimation:
        """
        最新の試行結果からアニメーションを作成する．

        Parameters
        ----------
        dt : float
            1フレームの時間
        n_frames : int
            総フレーム数

        Returns
        -------
        ani : animation.FuncAnimation
            アニメーションオブジェクト
        """
        assert dt > 0.
        assert n_frames > 0

        # 初期設定
        anim_fig = plt.figure()
        axis = anim_fig.add_subplot(111)
        axis.set_aspect('equal', adjustable='box')
        self._set_imgs(axis)

        # アニメーションを作成
        interval = dt * 1000.
        ani = animation.FuncAnimation(
            anim_fig,
            self._plot_func,
            interval=interval,
            frames=n_frames,
        )
        return ani

    @final
    def save_animation(self, save_path: str, dt: float, n_frames: int) -> None:
        """
        最新の試行結果をアニメーションとして保存する．

        Parameters
        ----------
        save_path : str
            保存先のパス．mp4形式．
        dt : float
            1フレームの時間
        n_frames : int
            総フレーム数
        """
        assert save_path.endswith('.mp4')
        assert dt > 0.
        assert n_frames > 0

        # アニメーションを作成
        ani = self.get_animation(dt, n_frames)

        # アニメーションを保存
        fps = 1. / dt
        writer = animation.writers['ffmpeg'](fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)
