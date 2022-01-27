from abc import *
from itertools import cycle

from exceptions import InvalidMoveException

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

    @abstractmethod
    def move(self, move: str, inplace: bool = True) -> 'board':
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
    Players in a game will all have a reference to the same board.
    This allows for a better synchronization between all players.
    """
    def __init__(self, _board: board) -> None:
        super().__init__()
        self.board = _board

    @abstractmethod
    def move(self) -> dict:
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

class two_player_game():
    """
    This class fully handles running a game where players take turns in order.
    """
    def __init__(self, _board: board, player_1: player, player_2: player) -> None:
        self.player_1 = player_1
        self.player_2 = player_2
        self.board = _board

    def play(self) -> dict:
        """
        This function cycles through players until a terminal board state is reached.
        """
        
        # Start by initializing the game
        self.board.reset()
        board_list = [self.board.copy()]
        it = iter(cycle([self.player_1, self.player_2]))
        current_player = next(it)

        while not self.board.is_terminal:
            # Have the player choose a move
            _move = current_player.move()["move_code"]

            # Try to make the chosen move
            try:
                self.board.move(_move)
            except InvalidMoveException as e:
                print(*e.args)
                if input("Try again? [y/n]") == 'y':
                    continue

            # If the move was successful then add the new board state to the list
            board_list.append(self.board.copy())
            
            # Move to the next player
            current_player = next(it)

            # Inform them of the move made by the other player
            current_player.inform(_move)

        # Now the game is finished
        # I'm leaving this as a dictionary just because idk if in the future I'd like to report more info
        # Or, have optionally different types of returns based on configuration settings.
        return {
            "boards": board_list
        }