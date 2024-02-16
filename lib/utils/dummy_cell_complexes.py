import torch

from lib.data.cochain import Cochain
from lib.data.complex import Complex

def get_house_complex():
    """
    Returns the `house graph` below with dummy features.
    The `house graph` (3-2-4 is a filled triangle):
       4
      / \
     3---2
     |   |
     0---1

       .
      4 5
     . 2 .
     3   1
     . 0 .

       .
      /0\
     .---.
     |   |
     .---.
    """
    v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4],                        # index of upper adjacencies (neighbors sharing co-boundary)
                               [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3]], dtype=torch.long)
    # each index is the index of the coboundary i.e. edge where vertex is face of.
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2, 5, 5, 4, 4], dtype=torch.long) 
    v_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)            # dummy features
    yv = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)                        # dummy labels
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3], [3, 4], [2, 4]]                              # index of boundaries (two vertices of edge)
    e_boundary_index = torch.stack([                                                             # flatten so that 1st row is the index of the boundary, second row is the index of the cochain.
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).view(-1)], 0)

    e_up_index = torch.tensor([[2, 4, 2, 5, 4, 5],                                  # index of upper adjacencies  (neighbors sharing co-boundary)
                               [4, 2, 5, 2, 5, 4]], dtype=torch.long)
    e_shared_coboundaries = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)      # each index is the index of the coboundary i.e. triangle where edge is face of.
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 4, 5],    # index of lower adjacencies (neighbors sharing boundary)
                                 [1, 0, 3, 0, 2, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 4]],
        dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 4, 4],  # each index is the index of the boundary i.e. vertex which is face of the edge.
        dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)       # dummy features
    ye = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)                     # dummt labels
    e_cochain = Cochain(dim=1, x=e_x, upper_index=e_up_index, lower_index=e_down_index,
        shared_coboundaries=e_shared_coboundaries, shared_boundaries=e_shared_boundaries,
        boundary_index=e_boundary_index, y=ye)

    t_boundaries = [[2, 4, 5]]                                      # boundary of triangle 0
    t_boundary_index = torch.stack([                                # first row is edge, second row is triangle
        torch.LongTensor(t_boundaries).view(-1),
        torch.LongTensor([0, 0, 0]).view(-1)], 0)
    # lack of t_down_index and t_shared_boundaries since there is only one triangle.
    # the highest cochain doesn't have shared_coboundaries or up_index
    t_x = torch.tensor([[1]], dtype=torch.float)
    yt = torch.tensor([2], dtype=torch.long)
    t_cochain = Cochain(dim=2, x=t_x, y=yt, boundary_index=t_boundary_index)

    
    y = torch.LongTensor([v_x.shape[0]])
    return Complex(v_cochain, e_cochain, t_cochain, y=y)


def get_bridged_complex():
    """
    Returns the `bridged graph` below with dummy features.
    The `bridged graph` (0-1-4-3, 1-2-3-4, 0-1-2-3 are filled rings): 
      
     3---2
     |\  |  
     | 4 |
     |  \|
     0---1

     .-2-.
     |4  |  
     3 . 1
     |  5|
     .-0-.

     .---.
     |\1 |  
     | . |
     | 0\|
     .---.
     
     .---.
     |   |  
     | 2 |
     |   |
     .---.
    """
    v_up_index = torch.tensor(     [[0, 1, 0, 3, 1, 2, 1, 4, 2, 3, 3, 4],
                                    [1, 0, 3, 0, 2, 1, 4, 1, 3, 2, 4, 3]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 5, 5, 2, 2, 4, 4], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3], [3, 4], [1, 4]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).view(-1)], 0)

    e_up_index = torch.tensor(     [[0, 1, 0, 2, 0, 3, 0, 3, 0, 4, 0, 5, 1, 2, 1, 2, 1, 3, 1, 4, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 3, 5, 4, 5, 4, 5],
                                    [1, 0, 2, 0, 3, 0, 3, 0, 4, 0, 5, 0, 2, 1, 2, 1, 3, 1, 4, 1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 3, 5, 4, 5, 4]], dtype=torch.long)
    e_shared_coboundaries = torch.tensor([2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.long)
    
    e_down_index = torch.tensor( [[0, 1, 0, 3, 0, 5, 1, 2, 1, 5, 2, 3, 2, 4, 3, 4, 4, 5],
                                  [1, 0, 3, 0, 5, 0, 2, 1, 5, 1, 3, 2, 4, 2, 4, 3, 5, 4]], dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 3, 3, 3, 3, 3, 3, 4, 4], dtype=torch.long)
    
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    e_cochain = Cochain(dim=1, x=e_x, upper_index=e_up_index, lower_index=e_down_index,
        shared_coboundaries=e_shared_coboundaries, shared_boundaries=e_shared_boundaries,
        boundary_index=e_boundary_index, y=ye)
    
    t_boundaries = [[0, 3, 4, 5], [1, 2, 4, 5], [0, 1, 2, 3]]
    t_boundary_index = torch.stack([
        torch.LongTensor(t_boundaries).view(-1),
        torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]).view(-1)], 0)
    t_down_index = torch.tensor( [[0, 1, 0, 1, 0, 2, 0, 2, 1, 2, 1, 2],
                                  [1, 0, 1, 0, 2, 0, 2, 0, 2, 1, 2, 1]], dtype=torch.long)
    t_shared_boundaries = torch.tensor([4, 4, 5, 5, 0, 0, 3, 3, 1, 1, 2, 2], dtype=torch.long)
    t_x = torch.tensor([[1], [2], [3]], dtype=torch.float)
    yt = torch.tensor([2, 2, 2], dtype=torch.long)
    t_cochain = Cochain(dim=2, x=t_x, y=yt, boundary_index=t_boundary_index, lower_index=t_down_index, shared_boundaries=t_shared_boundaries)

    y = torch.LongTensor([v_x.shape[0]])

    return Complex(v_cochain, e_cochain, t_cochain, y=y)


def get_fullstop_complex():
    """
    Returns the `fullstop graph` below with dummy features.
    The `fullstop graph` is a single isolated node:

    0

    """
    v_x = torch.tensor([[1]], dtype=torch.float)
    yv = torch.tensor([0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, y=yv)
    y = torch.LongTensor([v_x.shape[0]])
    return Complex(v_cochain, y=y)


def get_colon_complex():
    """
    Returns the `colon graph` below with dummy features.
    The `colon graph` is made up of two isolated nodes:

    1

    0

    """
    v_x = torch.tensor([[1], [2]], dtype=torch.float)
    yv = torch.tensor([0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, y=yv)
    y = torch.LongTensor([v_x.shape[0]])
    return Complex(v_cochain, y=y)


def get_square_complex():
    """
    Returns the `square graph` below with dummy features.
    The `square graph`:

     3---2
     |   |
     0---1

     . 2 .
     3   1
     . 0 .

     .---.
     |   |
     .---.
    """
    v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                               [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3]).view(-1)], 0)
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                                 [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 3, 3], dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, shared_boundaries=e_shared_boundaries, y=ye,
        boundary_index=e_boundary_index)
    
    y = torch.LongTensor([v_x.shape[0]])
    
    return Complex(v_cochain, e_cochain, y=y)


def get_square_dot_complex():
    """
    Returns the `square-dot graph` below with dummy features.
    The `square-dot graph`:

     3---2
     |   |
     0---1  4

     . 2 .
     3   1
     . 0 .  .

     .---.
     |   |
     .---.  .
    """
    v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                               [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3]).view(-1)], 0)
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                                 [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 3, 3], dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, shared_boundaries=e_shared_boundaries, y=ye,
        boundary_index=e_boundary_index)
    
    y = torch.LongTensor([v_x.shape[0]])

    return Complex(v_cochain, e_cochain, y=y)


def get_kite_complex():
    """
    Returns the `kite graph` below with dummy features.
    The `kite graph`:

      2---3---4
     / \ /
    0---1

      . 4 . 5 .
     2 1 3
    . 0 .

      .---.---.
     /0\1/
    .---.
    
    """
    v_up_index = torch.tensor([[0, 1, 0, 2, 1, 2, 1, 3, 2, 3, 3, 4],
                               [1, 0, 2, 0, 2, 1, 3, 1, 3, 2, 4, 3]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 2, 2, 1, 1, 3, 3, 4, 4, 5, 5], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [0, 2], [1, 3], [2, 3], [3, 4]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).view(-1)], 0)

    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 3, 0, 2, 1, 2, 2, 4, 1, 4, 3, 4, 3, 5, 4, 5],
                                 [1, 0, 3, 0, 3, 1, 2, 0, 2, 1, 4, 2, 4, 1, 4, 3, 5, 3, 5, 4]],
        dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
        dtype=torch.long)
    e_up_index = torch.tensor([[0, 1, 0, 2, 1, 2, 1, 3, 1, 4, 3, 4],
                               [1, 0, 2, 0, 2, 1, 3, 1, 4, 1, 4, 3]], dtype=torch.long)
    e_shared_coboundaries = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)

    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, shared_boundaries=e_shared_boundaries,
        upper_index=e_up_index, shared_coboundaries=e_shared_coboundaries, y=ye,
        boundary_index=e_boundary_index)

    t_boundaries = [[0, 1, 2], [1, 3, 4]]
    t_boundary_index = torch.stack([
        torch.LongTensor(t_boundaries).view(-1),
        torch.LongTensor([0, 0, 0, 1, 1, 1]).view(-1)], 0)

    t_down_index = torch.tensor([[0, 1],
                                 [1, 0]], dtype=torch.long)
    t_shared_boundaries = torch.tensor([1, 1], dtype=torch.long)
    t_x = torch.tensor([[1], [2]], dtype=torch.float)
    yt = torch.tensor([2, 2], dtype=torch.long)
    t_cochain = Cochain(dim=2, x=t_x, lower_index=t_down_index, shared_boundaries=t_shared_boundaries, y=yt,
        boundary_index=t_boundary_index)

    y = torch.LongTensor([v_x.shape[0]])

    return Complex(v_cochain, e_cochain, t_cochain, y=y)


def get_pyramid_complex():
    """
    Returns the `pyramid` below with dummy features.
    The `pyramid` (corresponds to a 4-clique):
    
       3
      /|\
     /_2_\
    0-----1

       .
     5 4 3
      2.1
    .  0  .
    
       3
      / \
     /   \
    2-----1
   / \   / \
  /   \ /   \
 3-----0-----3
    
       .
      / \
     4   3
    .--1--.
   / 2   0 \
  4   \ /   3
 .--5--.--5--.
    
       3
      / \
     / 2 \
    2-----1
   / \ 0 / \
  / 3 \ / 1 \
 3-----0-----3
 
       .
      /|\
     /_0_\
    .-----.
  
  """
    v_up_index = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                               [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 2, 2, 5, 5, 1, 1, 3, 3, 4, 4], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    yv = torch.tensor([3, 3, 3, 3], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [0, 2], [1, 3], [2, 3], [0, 3]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).view(-1)], 0)

    e_up_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2, 0, 5, 0, 3, 3, 5, 1, 3, 1, 4, 3, 4, 2, 4, 2, 5, 4, 5],
         [1, 0, 2, 0, 2, 1, 5, 0, 3, 0, 5, 3, 3, 1, 4, 1, 4, 3, 4, 2, 5, 2, 5, 4]],
        dtype=torch.long)
    e_shared_coboundaries = torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3], dtype=torch.long)
    e_down_index = torch.tensor(
        [[0, 1, 0, 2, 0, 3, 0, 5, 1, 2, 1, 3, 1, 4, 2, 4, 2, 5, 3, 4, 3, 5, 4, 5],
         [1, 0, 2, 0, 3, 0, 5, 0, 2, 1, 3, 1, 4, 1, 4, 2, 5, 2, 4, 3, 5, 3, 5, 4]],
        dtype=torch.long)
    e_shared_boundaries = torch.tensor(
        [1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 1, 1, 2, 2, 2, 2, 0, 0, 3, 3, 3, 3, 3, 3], dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, upper_index=e_up_index,
        shared_boundaries=e_shared_boundaries, shared_coboundaries=e_shared_coboundaries, y=ye,
        boundary_index=e_boundary_index)

    t_boundaries = [[0, 1, 2], [0, 3, 5], [1, 3, 4], [2, 4, 5]]
    t_boundary_index = torch.stack([
        torch.LongTensor(t_boundaries).view(-1),
        torch.LongTensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]).view(-1)], 0)

    t_up_index = torch.tensor([[0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3],
                               [1, 0, 2, 0, 2, 1, 3, 0, 3, 1, 3, 2]], dtype=torch.long)
    t_shared_coboundaries = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    t_down_index = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                                 [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    t_shared_boundaries = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 5, 5, 4, 4], dtype=torch.long)
    t_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    yt = torch.tensor([2, 2, 2, 2], dtype=torch.long)
    t_cochain = Cochain(dim=2, x=t_x, lower_index=t_down_index, upper_index=t_up_index,
        shared_boundaries=t_shared_boundaries, shared_coboundaries=t_shared_coboundaries, y=yt,
        boundary_index=t_boundary_index)

    p_boundaries = [[0, 1, 2, 3]]
    p_boundary_index = torch.stack([
        torch.LongTensor(p_boundaries).view(-1),
        torch.LongTensor([0, 0, 0, 0]).view(-1)], 0)
    p_x = torch.tensor([[1]], dtype=torch.float)
    yp = torch.tensor([3], dtype=torch.long)
    p_cochain = Cochain(dim=3, x=p_x, y=yp, boundary_index=p_boundary_index)

    y = torch.LongTensor([v_x.shape[0]])
        
    return Complex(v_cochain, e_cochain, t_cochain, p_cochain, y=y)


def get_filled_square_complex():
    """This is a cell / cubical complex formed of a single filled square.

     3---2
     |   |
     0---1

     . 2 .
     3   1
     . 0 .

     .---.
     | 0 |
     .---.
    """

    v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                               [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 2, 2], dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3]).view(-1)], 0)
    e_down_index = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3],
                                 [1, 0, 3, 0, 2, 1, 3, 2]], dtype=torch.long)
    e_shared_boundaries = torch.tensor([1, 1, 0, 0, 2, 2, 3, 3], dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1], dtype=torch.long)

    e_upper_index = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                                  [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    e_shared_coboundaries = torch.tensor([0]*12, dtype=torch.long)

    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, shared_boundaries=e_shared_boundaries,
        upper_index=e_upper_index, y=ye, shared_coboundaries=e_shared_coboundaries, boundary_index=e_boundary_index)

    c_boundary_index = torch.LongTensor(
        [[0, 1, 2, 3],
         [0, 0, 0, 0]]
    )
    c_x = torch.tensor([[1]], dtype=torch.float)
    yc = torch.tensor([2], dtype=torch.long)
    c_cochain = Cochain(dim=2, x=c_x, y=yc, boundary_index=c_boundary_index)
    
    y = torch.LongTensor([v_x.shape[0]])

    return Complex(v_cochain, e_cochain, c_cochain, y=y)


def get_molecular_complex():
    """This is a molecule with filled rings.

     3---2---4---5
     |   |       |
     0---1-------6---7

     . 2 . 4 . 5 .
     3   1       6
     . 0 .   7   . 8 .

     .---. --- . --- .
     | 0 |    1      |
     .---. --------- . ---- .
    """

    v_up_index = torch.tensor([[0, 1, 0, 3, 1, 2, 1, 6, 2, 3, 2, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 3, 0, 2, 1, 6, 1, 3, 2, 4, 2, 5, 4, 6, 5, 7, 6]],
        dtype=torch.long)
    v_shared_coboundaries = torch.tensor([0, 0, 3, 3, 1, 1, 7, 7, 2, 2, 4, 4, 5, 5, 6, 6, 8, 8],
        dtype=torch.long)
    v_x = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float)
    yv = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    v_cochain = Cochain(dim=0, x=v_x, upper_index=v_up_index, shared_coboundaries=v_shared_coboundaries, y=yv)

    e_boundaries = [[0, 1], [1, 2], [2, 3], [0, 3], [1, 6], [2, 4], [4, 5], [5, 6], [6, 7]]
    e_boundary_index = torch.stack([
        torch.LongTensor(e_boundaries).view(-1),
        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 7, 7, 4, 4, 5, 5, 6, 6, 8, 8]).view(-1)], 0)
    e_down_index = torch.tensor(
        [[0, 1, 0, 3, 1, 2, 2, 3, 1, 4, 2, 4, 4, 5, 5, 6, 6, 7, 6, 8, 7, 8, 0, 7, 1, 7],
         [1, 0, 3, 0, 2, 1, 3, 2, 4, 1, 4, 2, 5, 4, 6, 5, 7, 6, 8, 6, 8, 7, 7, 0, 7, 1]],
        dtype=torch.long)
    e_shared_boundaries = torch.tensor(
        [1, 1, 0, 0, 2, 2, 3, 3, 2, 2, 2, 2, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1],
        dtype=torch.long)
    e_x = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=torch.float)
    ye = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)

    e_upper_index_c1 = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                                     [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    e_upper_index_c2 = torch.tensor([[1, 4, 1, 5, 1, 6, 1, 7, 4, 5, 4, 6, 4, 7, 5, 6, 5, 7, 6, 7],
                                     [4, 1, 5, 1, 6, 1, 7, 1, 5, 4, 6, 4, 7, 4, 6, 5, 7, 5, 7, 6]],
        dtype=torch.long)
    e_upper_index = torch.cat((e_upper_index_c1, e_upper_index_c2), dim=-1)
    e_shared_coboundaries = torch.tensor([0]*12 + [1]*20, dtype=torch.long)

    e_cochain = Cochain(dim=1, x=e_x, lower_index=e_down_index, shared_boundaries=e_shared_boundaries,
        upper_index=e_upper_index, y=ye, shared_coboundaries=e_shared_coboundaries, boundary_index=e_boundary_index)

    c_boundary_index = torch.LongTensor(
        [[0, 1, 2, 3, 1, 4, 5, 6, 7],
         [0, 0, 0, 0, 1, 1, 1, 1, 1]]
    )
    c_x = torch.tensor([[1], [2]], dtype=torch.float)
    c_down_index = torch.tensor([[0, 1],
                                 [1, 0]], dtype=torch.long)
    c_shared_boundaries = torch.tensor([1, 1],  dtype=torch.long)

    yc = torch.tensor([2, 2], dtype=torch.long)
    c_cochain = Cochain(dim=2, x=c_x, y=yc, boundary_index=c_boundary_index, lower_index=c_down_index,
        shared_boundaries=c_shared_boundaries)
    
    y = torch.LongTensor([v_x.shape[0]])

    return Complex(v_cochain, e_cochain, c_cochain, y=y)