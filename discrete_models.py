import torch
from torch import Tensor, optim, nn
from typing import Iterable, Callable
import matplotlib.pyplot as plt
import itertools
import IPython
from interruptable_code import Interruptable


class DiscreteGamePlayer(nn.Module):
    """Represents a player in a discrete game"""
    def __init__(self, my_move_size: int, opponent_move_size: int):
        super().__init__()
        self.my_move_size: int = my_move_size
        self.opponent_move_size: int = opponent_move_size
        self.payoff_matrix = self.get_payoff_matrix()
        self.p = nn.Parameter(torch.rand(my_move_size, requires_grad=True))

    @property
    def pmf(self) -> Tensor:
        """
		The pmf of the discrete game player
		"""
        # return nn.Softmax()(self.p)
        return torch.abs(self.p) / torch.sum(torch.abs(self.p))

    def payout(self, move1: int, move2: int) -> float:
        """
        Represents self's payoff if self plays move1 and the opponent plays move2
        """
        raise NotImplementedError()

    def get_payoff_matrix(self) -> Tensor:
        """
        Returns the payoff matrix of self
        """
        payoff_matrix = torch.zeros((self.my_move_size, self.opponent_move_size))
        for i in range(self.my_move_size):
            for j in range(self.opponent_move_size):
                payoff_matrix[i, j] = self.payout(i, j)
        return payoff_matrix

    def forward(self, opponent_pmf: Tensor):
        """
        Returns the expected payout of self given the opponent's parameter
        """
        return (
            torch.matmul(self.pmf.reshape(-1, 1), opponent_pmf.reshape(1, -1))
            * self.payoff_matrix
        ).sum()

    def max_payout(self, opponent_pmf: Tensor) -> Tensor:
        """
        Returns the maximum expected payout of self across all possible self's strategies given the opponent's pmf
        """
        # We have a payout matrix. We basically want to scale all the j's with the
        # opponents strategy, take the sum from axis = 1, and return the max item in this structure
        all_payoffs = (opponent_pmf * self.payoff_matrix).sum(axis=1)
        return all_payoffs.max()


class DiscreteIterativeTrainer:
    """
    This keeps iteratively updating the players strategy in the direction of most improvement
    """

    OptimizerFactory = Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]

    def __init__(
        self,
        player1: DiscreteGamePlayer,
        player2: DiscreteGamePlayer,
        optim: OptimizerFactory = optim.Adam,
    ):
        if (player1.my_move_size != player2.opponent_move_size
        or player1.opponent_move_size != player2.my_move_size):
            raise ValueError("Player 1 and Player 2's move sizes must be compatible")
        self.player1: DiscreteGamePlayer = player1
        self.player2: DiscreteGamePlayer = player2
        self.optim1: optim.Optimizer = optim(self.player1.parameters())
        self.optim2: optim.Optimizer = optim(self.player2.parameters())
        self.epoch = 0

    def train(
        self,
        limit=10000,
        plot_epoch=200,
        action_labels_1=None,
        action_labels_2=None,
        only_player: int = None,
    ):
        if action_labels_1 is None:
            action_labels_1 = [i for i in range(self.player1.my_move_size)]
        if action_labels_2 is None:
            action_labels_2 = [i for i in range(self.player2.my_move_size)]
        # Starting the interactive gradient descent
        with Interruptable() as check_interrupted:
            while self.epoch <= limit:
                check_interrupted()
                if only_player != 2:
                    self.optim1.zero_grad()
                    payoff1: Tensor = self.player1(self.player2.pmf.detach())
                    (-payoff1).backward()
                    self.optim1.step()
                if only_player != 1:
                    self.optim2.zero_grad()
                    payoff2: Tensor = self.player2(self.player1.pmf.detach())
                    (-payoff2).backward()
                    self.optim2.step()
                if self.epoch % plot_epoch == 0:
                    IPython.display.clear_output(wait=True)
                    print(f"epoch = {self.epoch}")
                    # plotting the distributions
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    # Plot the first PDF
                    axes[0].bar(
                        action_labels_1,
                        self.player1.pmf.detach().numpy(),
                        label="PDF 1",
                    )
                    axes[0].set_title("Player 1")
                    # Plot the second PDF
                    axes[1].bar(
                        action_labels_2,
                        self.player2.pmf.detach().numpy(),
                        label="PDF 2",
                    )
                    axes[1].set_title("Player 2")
                    plt.show()
                self.epoch += 1


class DiscreteLossTrainer:
    """Uses a differentiable loss function to find the equilibrium"""

    OptimizerFactory = Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]

    def __init__(
        self,
        player1: DiscreteGamePlayer,
        player2: DiscreteGamePlayer,
        optim: OptimizerFactory = optim.Adam,
    ):
        if (
            player1.my_move_size != player2.opponent_move_size
            or player1.opponent_move_size != player2.my_move_size
        ):
            raise ValueError("Player 1 and Player 2's move sizes must be compatible")
        self.player1: DiscreteGamePlayer = player1
        self.player2: DiscreteGamePlayer = player2
        self.optim: optim.Optimizer = optim(
            itertools.chain(self.player1.parameters(), self.player2.parameters())
        )
        self.epoch = 0

    def get_loss(self) -> Tensor:
        """Returns a differentiable loss which represents the deviation from the nash equilibrium"""
        return (
            self.player1.max_payout(self.player2.pmf)
            - self.player1(self.player2.pmf)
            + self.player2.max_payout(self.player1.pmf)
            - self.player2(self.player1.pmf)
        )

    def train(
        self, limit=10000, plot_epoch=200, action_labels_1=None, action_labels_2=None
    ):
        if action_labels_1 is None:
            action_labels_1 = [i for i in range(self.player1.my_move_size)]
        if action_labels_2 is None:
            action_labels_2 = [i for i in range(self.player2.my_move_size)]
        # Starting the interactive gradient descent
        with Interruptable() as check_interrupted:
            while self.epoch <= limit:
                check_interrupted()
                self.optim.zero_grad()
                loss = self.get_loss()
                loss.backward()
                self.optim.step()
                if self.epoch % plot_epoch == 0:
                    IPython.display.clear_output(wait=True)
                    print(f"epoch = {self.epoch}")
                    print(f"loss = {loss.detach()}")
                    # plotting the distributions
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    # Plot the first PDF
                    axes[0].bar(
                        action_labels_1,
                        self.player1.pmf.detach().numpy(),
                        label="PDF 1",
                    )
                    axes[0].set_title("Player 1")
                    # Plot the second PDF
                    axes[1].bar(
                        action_labels_2,
                        self.player2.pmf.detach().numpy(),
                        label="PDF 2",
                    )
                    axes[1].set_title("Player 2")
                    plt.show()
                self.epoch += 1
