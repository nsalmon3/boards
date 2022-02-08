from abc import *
from itertools import cycle

from exceptions import InvalidMoveException

class bid(type):
    def __str__(self):
        if hasattr(self, 'as_string'):
            return self.as_string
        else:
            return 'bid_has_no_string'

_BASE_ELO = 400
_K = 30
class elo():
    def __init__(self, _elo: float = _BASE_ELO) -> None:
        self._elo = _elo
    
    def __str__(self):
        return str(self._elo)
    
    @property
    def q(self):
        return 10 ** (self._elo / _BASE_ELO)
    
    @classmethod
    def E(cls, elo_a: 'elo', elo_b: 'elo') -> float:
        return elo_a.q / (elo_a.q + elo_b.q)
    
    @classmethod
    def update(cls, elo_a: 'elo', elo_b: 'elo', s_a: float, s_b: float):
        elo_a._elo, elo_b._elo = elo_a._elo + _K * (s_a - cls.E(elo_a, elo_b)), elo_b._elo + _K * (s_b - cls.E(elo_b, elo_a))

class board(ABC):
    """
    This is the abstract board class defining how boards must be defined.
    """
    def __init__(self, _moves: 'list[bid]', *args, **kwargs):
        """
        The only logic in here is to allow derived classes to easily define properties upon initialization.
        """
        super().__init__()
        for arg in kwargs:
            self.__dict__[arg] = kwargs[arg]
        self._moves = _moves

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Checks to see if the board is in a terminal state.
        """
        ...

    @property
    def moves(self) -> 'list[bid]':
        """
        Returns the list of valid moves in the current board state.
        """
        return self._moves

    @abstractmethod
    def move(self, move: bid, inplace: bool = True) -> 'board':
        """
        Attempts to make a move, and returns the updated board after the move.
        inplace describes whether it is returning a reference to the same board or a new board.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the board to it's initial state.
        Used so that players can have a reference to the same board across multiple games.
        """
        ...

    @abstractmethod
    def copy(self) -> 'board':
        """
        Returns a deep copy of a board.
        """
        ...

    @abstractmethod
    def jsonify(self) -> dict:
        """
        A dict representation of the board used for compact storage of the board on disk.
        """
        ...

    @classmethod
    @abstractmethod
    def dejsonify(cls, ser: str) -> 'board':
        """
        Takes in a string representation of the board, defined by the serialize function, and returns the corresponding board.
        """
        ...

class player(ABC):
    """
    This is the abstract player class defining how all other players must be defined.
    Players in a game will all have a reference to the same board.
    This allows for a better synchronization between all players.
    """
    def __init__(self, _board: board, _elo: elo = None) -> None:
        super().__init__()
        self.board = _board
        if _elo is None:
            self.elo = elo()
        else:
            self.elo = _elo

    @abstractmethod
    def move(self) -> bid:
        """
        The logic that decides what move to make. The primary difference between players is contained here.
        """
        ...

    @abstractmethod
    def inform(self, _move: bid):
        """
        There are some players that need to know when other players make moves.
        A game can call this on each player every time a move is made, so that players get updates on other players moves.
        """
        ...
    
    @abstractmethod
    def reset(self):
        """
        Like inform, some players need to do some things in between tournament games. We let them know to reset with this.
        """