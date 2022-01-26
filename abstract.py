from abc import *

class board(ABC):
    """
    This is the abstract board class defining how boards must be defined.
    """
    def __init__(self, *args, **kwargs):
        """
        The only logic in here is to allow derived classes to easily define properties upon initialization.
        """
        for arg in kwargs:
            self.__dict__[arg] = kwargs[arg]

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Checks to see if the board is in a terminal state.
        """
        ...

    @property
    @abstractmethod
    def moves(self) -> 'list[str]':
        """
        Returns the list of valid moves in the current board state.
        """
        ...

    @classmethod
    @abstractmethod
    def move(cls, _board: 'board', move: str, inplace: bool = False) -> 'board':
        """
        Attempts to make a move, and returns the updated board after the move.
        If inplace is true, it should return a reference to the same board, only updated.
        """
        ...

    @classmethod
    @abstractmethod
    def copy(cls, _board: 'board') -> 'board':
        """
        Returns a deep copy of a board.
        """
        ...

    @abstractmethod
    def serialize(self) -> str:
        """
        A string representation of the board used for compact storage of the board on disk.
        """
        ...

    @classmethod
    @abstractmethod
    def deserialize(cls, ser: str) -> 'board':
        """
        Takes in a string representation of the board, defined by the serialize function, and returns the corresponding board.
        """
        ...

class player(ABC):
    """
    This is the abstract player class defining how all other players must be defined.
    """
    @abstractmethod
    def move(self, _board: board) -> str:
        """
        The logic that decides what move to make. The primary difference between players is contained here.
        """
        ...

    @abstractmethod
    def inform(self, _move: str):
        """
        There are some players that need to know when other players make moves.
        A game can call this on each player every time a move is made, so that players get updates on other players moves.
        """
        ...

class game(ABC):
    """
    This is the abstract game class defining how all other games must be defined.
    """
    @abstractmethod
    def play(self) -> dict:
        """
        The logic for playing out a game. Returns a dictionary that is meant to summarize the game after it ends.
        """
        ...