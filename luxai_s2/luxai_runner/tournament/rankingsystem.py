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

    def update(
        self, rank_1: Rank, rank_2: Rank, rank_1_score: float, rank_2_score: float
    ):
        pass

    def _rank_headers(self) -> str:
        pass

    def _rank_info(self, rank: Rank) -> str:
        pass


@dataclass
class ELORank(Rank):
    rating: int
    episodes: int


class ELO(RankingSystem):
    def __init__(self, K=32, init_rating=1000, col_length=10) -> None:
        super().__init__()
        self.col_length = col_length
        self.K = K
        self.init_rating = 1000

    def init_rank_state(self) -> Rank:
        return ELORank(rating=self.init_rating, episodes=0)

    def update(self, rank_1: ELORank, rank_2: ELORank, rank_1_score, rank_2_score):
        # Only implements win/loss/ties
        if rank_1_score > rank_2_score:
            rank_1.rating = rank_1.rating + self.K * (
                1 - self._expected_score(rank_1, rank_2)
            )
            rank_2.rating = rank_2.rating + self.K * (
                0 - self._expected_score(rank_2, rank_1)
            )
        elif rank_1_score < rank_2_score:
            rank_1.rating = rank_1.rating + self.K * (
                0 - self._expected_score(rank_1, rank_2)
            )
            rank_2.rating = rank_2.rating + self.K * (
                1 - self._expected_score(rank_2, rank_1)
            )
        else:
            rank_1.rating = rank_1.rating + self.K * (
                0.5 - self._expected_score(rank_1, rank_2)
            )
            rank_2.rating = rank_2.rating + self.K * (
                0.5 - self._expected_score(rank_2, rank_1)
            )
        rank_1.episodes += 1
        rank_2.episodes += 1

    def _expected_score(self, rank_1: ELORank, rank_2: ELORank):
        """
        Returns E_1 = expected score for player 1
        """
        return 1 / (1 + np.power(10, (rank_2.rating - rank_1.rating) / 400))

    def _rank_headers(self) -> str:
        return f"{'Rating':{self.col_length}.{self.col_length}}"

    def _rank_info(self, rank: ELORank) -> str:
        return f"{str(rank.rating):{self.col_length}.{self.col_length}}"


@dataclass
class WinLossRank(Rank):
    rating: int
    wins: int
    ties: int
    losses: int
    episodes: int


class WinLoss(RankingSystem):
    def __init__(
        self, win_points=3, tie_points=1, loss_points=0, col_length=10
    ) -> None:
        super().__init__()
        self.col_length = col_length
        self.win_points = win_points
        self.tie_points = tie_points
        self.loss_points = loss_points

    def init_rank_state(self) -> Rank:
        return WinLossRank(rating=0, episodes=0, wins=0, ties=0, losses=0)

    def update(
        self, rank_1: WinLossRank, rank_2: WinLossRank, rank_1_score, rank_2_score
    ):
        # Only implements win/loss/ties
        if rank_1_score == rank_2_score:
            winner = None
            loser = None
            rank_1.ties += 1
            rank_2.ties += 1
            rank_1.rating = rank_1.rating + self.tie_points
            rank_2.rating = rank_2.rating + self.tie_points
        else:
            winner = rank_1
            loser = rank_2
            if rank_1_score < rank_2_score:
                winner = rank_2
                loser = rank_1
            # not a tie
            winner.wins += 1
            loser.losses += 1
            winner.rating = winner.rating + self.win_points
            loser.rating = loser.rating + self.loss_points

        rank_1.episodes += 1
        rank_2.episodes += 1

    def _rank_headers(self) -> str:
        return f"{'Score':{self.col_length}} | {'Wins':{self.col_length}} | {'Ties':{self.col_length}} | {'Losses':{self.col_length}}"

    def _rank_info(self, rank: WinLossRank) -> str:
        return f"{str(rank.rating):{self.col_length}.{self.col_length}} | {str(rank.wins):{self.col_length}.{self.col_length}} | {str(rank.ties):{self.col_length}.{self.col_length}} | {str(rank.losses):{self.col_length}.{self.col_length}}"
