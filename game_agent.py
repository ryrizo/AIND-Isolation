"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
from __future__ import print_function
import random
import numpy as np
import distance_matrix

from copy import deepcopy
from copy import copy


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass
expected_game_board_size = 7
knight_matrix = distance_matrix.generate_knight_distance_matrix(expected_game_board_size)

def moves_matrix_score(game, player):
    """ Takes the static matrix of how many moves it takes a piece to get to
    any given sqaure on a board twice the size of a game board. Based on the game
    board size and player locations, sub-matrices are taken, subtracted, then
    multiplied by a matrix with 1's in available spaces and 0's in unavailable
    spaces. The sum of the values in the resulting matrix is considered to be
    which player has an advantage getting the remaining locations.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_loc = game.get_player_location(player)
    opp_loc = game.get_player_location(game.get_opponent(player))
    own_move_distances = distance_matrix.get_sub_distance_matrix(own_loc, game.width, knight_matrix)
    opp_move_distances = distance_matrix.get_sub_distance_matrix(opp_loc, game.width, knight_matrix)
    open_moves_matrix = distance_matrix.get_moves_left_matrix(game)
    diff_move_distances = np.subtract(own_move_distances,opp_move_distances)
    distance_diff_of_available = np.multiply(diff_move_distances,open_moves_matrix)

    return float(np.sum(distance_diff_of_available) * -1) # * own_moves - opp moves


def improved_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Start with known heuristic
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)

def mobility_multiplied_score(game, player):
    """ Improved score weighted on available moves of legal moves """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    # Sum the available moves for each legal move.
    own_mobility = float(sum([len(game.__get_moves__(move)) for move in own_moves ]))
    opp_mobility = float(sum([len(game.__get_moves__(move)) for move in opp_moves ]))
    return (len(own_moves) * own_mobility) - (len(opp_moves) * opp_mobility)

# Selected Heuristic
def custom_score(game, player):
    """ Improved score adding moves available to legal moves """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_mobility = float(sum([len(game.__get_moves__(move)) for move in own_moves ]))
    opp_mobility = float(sum([len(game.__get_moves__(move)) for move in opp_moves ]))
    return own_mobility - opp_mobility + len(own_moves) - len(opp_moves)

def chase_score(game, player):
    """ Improved score with double opponent moves, incentivizes blocking """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - (opp_moves * 2))

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=10, score_fn=custom_score,
                 iterative=True, method='alphabeta', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if len(game.get_legal_moves(self)) == 0:
            return (-1,-1)

        if self.search_depth <= 0:
            self.iterative = True

        best_move = (None,(-1,-1))

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                for depth in range(1,game.width * game.height):
                    best_move = self.minimax(game, depth) if self.method == 'minimax' \
                        else self.alphabeta(game,depth)
                    if best_move[0] == float("inf"):
                        break
            else:
                best_move = self.minimax(game, self.search_depth) if self.method == 'minimax' \
                    else self.alphabeta(game, self.search_depth)

        except Timeout:
            return best_move[1]

        # Return the best move from the last completed search iteration
        return best_move[1]

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        scores_and_moves = set()
        player_perspective = self if maximizing_player else game.get_opponent(self)

        if len(game.get_legal_moves(player_perspective)) == 0:
            return (game.utility(self),(-1,-1))

        for move in game.get_legal_moves(player_perspective):
            score,child_move = (self.score(game.forecast_move(move),self),move) if depth == 1 \
                else self.minimax(game.forecast_move(move), depth-1, not maximizing_player)
            scores_and_moves.add((score,move))

        return max(scores_and_moves) if maximizing_player else min(scores_and_moves)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        scores_and_moves = set()
        player_perspective = self if maximizing_player else game.get_opponent(self)

        if len(game.get_legal_moves(player_perspective)) == 0:
            return (game.utility(self),(-1,-1))

        for move in game.get_legal_moves(player_perspective):
            score,child_move = (self.score(game.forecast_move(move), self), move) if depth == 1 \
                else self.alphabeta(game.forecast_move(move), depth-1, alpha, beta, not maximizing_player)
            if maximizing_player:
                if score >= beta:
                    # Maximizing player will choose this value or higher - min parent has lower option
                    return (score,move)
                # Set lower bound for next plie
                alpha = max(score, alpha)
            else:
                if score <= alpha:
                    # Minimizing player will choose this value or lower - max parent has higher option
                    return (score,move)
                # Set upper bound for next plie
                beta = min(score, beta)
            scores_and_moves.add((score,move))
        return max(scores_and_moves) if maximizing_player else min(scores_and_moves)
