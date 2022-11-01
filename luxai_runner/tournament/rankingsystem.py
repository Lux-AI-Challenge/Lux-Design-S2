from abc import ABC, abstractclassmethod
from dataclasses import dataclass
import numpy as np

class Rank:
    rating: int

class RankingSystem(ABC):
    def __init__(self) -> None:
        pass

    def init_rank_state(self) -> Rank:
        pass
    def update(self, rank_1: Rank, rank_2: Rank, rank_1_score: float, rank_2_score: float):
        pass


@dataclass
class ELORank(Rank):
    rating: int
    episodes: int
class ELO(RankingSystem):
    def __init__(self, K=32, init_rating=1000) -> None:
        super().__init__()
        self.K = K
        self.init_rating = 1000
    def init_rank_state(self) -> Rank:
        return ELORank(rating=self.init_rating, episodes=0)
    def update(self, rank_1: ELORank, rank_2: ELORank, rank_1_score, rank_2_score):
        # Only implements win/loss/ties
        if rank_1_score > rank_2_score:
            rank_1.rating = rank_1.rating + self.K * (1 - self._expected_score(rank_1, rank_2))
            rank_2.rating = rank_2.rating + self.K * (0 - self._expected_score(rank_2, rank_1))
        elif rank_1_score < rank_2_score:
            rank_1.rating = rank_1.rating + self.K * (0 - self._expected_score(rank_1, rank_2))
            rank_2.rating = rank_2.rating + self.K * (1 - self._expected_score(rank_2, rank_1))
        else:
            rank_1.rating = rank_1.rating + self.K * (0.5 - self._expected_score(rank_1, rank_2))
            rank_2.rating = rank_2.rating + self.K * (0.5 - self._expected_score(rank_2, rank_1))
        rank_1.episodes += 1
        rank_2.episodes += 1

    def _expected_score(self, rank_1: ELORank, rank_2: ELORank):
        """
        Returns E_1 = expected score for player 1
        """
        return 1 / (1 + np.power(10, (rank_2.rating - rank_1.rating) / 400))

@dataclass
class WinLossRank(Rank):
    rating: int
    episodes: int

class WinLoss(RankingSystem):
    def __init__(self, win_points=3,tie_points=1,loss_points=0) -> None:
        super().__init__()
        self.win_points=win_points
        self.tie_points=tie_points
        self.loss_points=loss_points
    def init_rank_state(self) -> Rank:
        return ELORank(rating=0, episodes=0)
    def update(self, rank_1: ELORank, rank_2: ELORank, rank_1_score, rank_2_score):
        # Only implements win/loss/ties
        if rank_1_score > rank_2_score:
            rank_1.rating = rank_1.rating + self.win_points
            rank_2.rating = rank_2.rating + self.loss_points
        elif rank_1_score < rank_2_score:
            rank_1.rating = rank_1.rating + self.loss_points
            rank_2.rating = rank_2.rating + self.win_points
        else:
            rank_1.rating = rank_1.rating + self.tie_points
            rank_2.rating = rank_2.rating + self.tie_points
        rank_1.episodes += 1
        rank_2.episodes += 1