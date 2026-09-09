from mrpp_navigation.astar import astar_grid


def test_astar_grid_finds_path_around_obstacle() -> None:
    obstacles = {(1, 0), (1, 1)}
    path = astar_grid(start=(0, 0), goal=(2, 0), obstacles=obstacles, width=4, height=4)

    assert path[0] == (0, 0)
    assert path[-1] == (2, 0)
    assert (1, 0) not in path
    assert (1, 1) not in path
