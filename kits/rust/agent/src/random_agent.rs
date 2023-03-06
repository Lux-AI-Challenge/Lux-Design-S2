use log::debug;
use lux::prelude::*;
use lux::{
    action::{Direction, ResourceType},
    utils::RectMat,
    Faction, Robot, RobotActionCommand, SetupAction, State, UnitAction, UnitActions,
};
use rand::{distributions::Uniform, rngs::SmallRng, Rng, SeedableRng};
use std::collections::BTreeSet;

pub struct RandomAgent {
    rng: SmallRng,
}

impl Default for RandomAgent {
    fn default() -> Self {
        let rng = SmallRng::from_entropy();
        Self { rng }
    }
}

impl RandomAgent {
    pub fn from_seed(seed: u64) -> Self {
        let rng = <SmallRng as SeedableRng>::seed_from_u64(seed);
        Self { rng }
    }
}

/// Builds a board array with each x, y index retrieving the position of the
/// closest `true` result in the mask_board if it is possible to move on a path
/// else the position will be `None`
///
/// # Notes
///
/// - Assumes that the all boolean vectors are the same length (ie a rectangular
/// 2d array), additionally assumes `mask_board` and `obstruction_board` are the
/// same length
/// - The implementation solves same distance equality by prioritising the
/// position with the lowest row-major coordinate
fn build_closest_board(
    mask_board: &RectMat<bool>,
    obstruction_board: Option<&RectMat<bool>>,
) -> RectMat<Option<Pos>> {
    let positions = mask_board
        .enumerate()
        .filter(|(_, val)| **val)
        .map(|(idx, _)| Pos(idx.0 as i64, idx.1 as i64))
        .collect::<Vec<_>>();
    let m = mask_board.r_len();
    let n = mask_board.c_len();
    let mut rv = RectMat::<Option<Pos>>::from_dims_default((m, n));
    let mut pos_queues = positions
        .into_iter()
        .map(|val| (val, BTreeSet::from([val])))
        .collect::<Vec<_>>();
    let deltas: Vec<Pos> = vec![Pos(0, 1), Pos(1, 0), Pos(0, -1), Pos(-1, 0)];
    while !pos_queues.is_empty() {
        pos_queues = pos_queues
            .into_iter()
            .map(|(val, curr_set)| {
                let next_set = curr_set
                    .iter()
                    .flat_map(|pos| {
                        if pos.0 < 0 || pos.1 < 0 || pos.0 >= m as i64 || pos.1 >= n as i64 {
                            return vec![];
                        }
                        let idx = pos.as_idx();
                        if let Some(obstruction_board) = obstruction_board {
                            if *obstruction_board.get(idx).unwrap() {
                                return vec![];
                            }
                        }
                        if rv.get_unchecked(idx).is_none() {
                            *rv.get_mut_unchecked(idx) = Some(val);
                            deltas.iter().map(|delta| *pos + *delta).collect::<Vec<_>>()
                        } else {
                            vec![]
                        }
                    })
                    .collect::<BTreeSet<_>>();
                (val, next_set)
            })
            .filter(|(_, next_set)| !next_set.is_empty())
            .collect::<Vec<_>>();
    }
    rv
}

/// Builds a board grid to answer the query for if there is an obstruction at
/// the given index.
///
/// ie `board.get(Pos(x,y))` being `true` would indicate there is an obstruction
fn build_obstruction_board(state: &State) -> RectMat<bool> {
    state
        .board
        .tiles
        .map_new(|tile| match &tile.factory_occupancy {
            Some(strain_id) => !state
                .my_team()
                .unwrap()
                .factory_strains
                .iter()
                .any(|x| x == strain_id),
            None => false,
        })
}

/// Builds a board array with each x, y index retrieving the position of the
/// closest ice patch from indexed coord if an ice patch is reachable
///
/// Returns `None` if there are no ice tiles reachable at the indexed coord
///
/// # Note
/// `obstruction_board` parameter available to avoid recomputation
fn build_closest_ice_map(
    state: &State,
    obstruction_board: Option<&RectMat<bool>>,
) -> RectMat<Option<Pos>> {
    let mask_board = state.board.tiles.map_new(|tile| tile.ice > 0);
    let tp_obstruction_board;
    let obstruction_board = {
        if let Some(rv) = obstruction_board {
            rv
        } else {
            tp_obstruction_board = build_obstruction_board(state);
            &tp_obstruction_board
        }
    };
    build_closest_board(&mask_board, Some(obstruction_board))
}

/// Builds a board array with each x, y index retrieving the position of the
/// closest factory tile owned by the player
///
/// # Note
/// `obstruction_board` parameter available to avoid recomputation
fn build_closest_self_factory_map(
    state: &State,
    obstruction_board: Option<&RectMat<bool>>,
) -> RectMat<Option<Pos>> {
    let mask_board = {
        let mut rv = RectMat::<bool>::from_dims_default((state.board.x_len(), state.board.y_len()));
        for factory in state.my_factories().values() {
            let (x_range, y_range) = factory.occupied_range();
            for x in x_range {
                for y in y_range.clone() {
                    let idx = Pos(x, y).as_idx();
                    *rv.get_mut_unchecked(idx) = true;
                }
            }
        }
        rv
    };
    let tp_obstruction_board;
    let obstruction_board = {
        if let Some(rv) = obstruction_board {
            rv
        } else {
            tp_obstruction_board = build_obstruction_board(state);
            &tp_obstruction_board
        }
    };
    build_closest_board(&mask_board, Some(obstruction_board))
}

/// Logic for a unit for a robot to gather ice up to 40 before returning to a factory
fn gather_ice_logic(
    closest_ice_map: &RectMat<Option<Pos>>,
    closest_self_factory_map: &RectMat<Option<Pos>>,
    robot: &Robot,
    state: &State,
) -> Option<UnitAction> {
    // note pathing does not account for obstructions, as such it is possible that a move towards
    // the closest destination won't be on the shortest path
    let idx = robot.pos.as_idx();
    if robot.cargo.ice < 40 {
        // move and dig
        closest_ice_map
            .get(idx)
            .unwrap()
            .as_ref()
            .and_then(|closest_ice_pos| {
                if *closest_ice_pos == robot.pos {
                    if robot.can_dig(state) {
                        Some(UnitAction::Robot(vec![RobotActionCommand::dig(None, None)]))
                    } else {
                        None
                    }
                } else {
                    let dir = Direction::move_towards(&robot.pos, closest_ice_pos);
                    if robot.can_move(state, &dir) {
                        Some(UnitAction::Robot(vec![RobotActionCommand::move_(
                            dir, None, None,
                        )]))
                    } else {
                        None
                    }
                }
            })
    } else {
        // get to a factory
        closest_self_factory_map
            .get(idx)
            .unwrap()
            .as_ref()
            .and_then(|closest_self_factory_pos| {
                let transfer_dir = {
                    let mut rv = None;
                    for direction in Direction::iter_all() {
                        let new_pos = robot.pos + direction.to_pos();
                        if new_pos == *closest_self_factory_pos {
                            rv = Some(direction);
                            break;
                        }
                    }
                    rv
                };
                if let Some(transfer_dir) = transfer_dir {
                    if robot.can_transfer(state) {
                        Some(UnitAction::Robot(vec![RobotActionCommand::transfer(
                            transfer_dir,
                            ResourceType::Ice,
                            robot.cargo.ice,
                            None,
                            None,
                        )]))
                    } else {
                        None
                    }
                } else {
                    let dir = Direction::move_towards(&robot.pos, closest_self_factory_pos);
                    if robot.can_move(state, &dir) {
                        Some(UnitAction::Robot(vec![RobotActionCommand::move_(
                            dir, None, None,
                        )]))
                    } else {
                        None
                    }
                }
            })
    }
}

/// An implementation mirroring the logic of the python kit's agent with some slightly
/// better path computation logic
impl Agent for RandomAgent {
    fn setup(&mut self, state: &State) -> Option<SetupAction> {
        debug!("requested setup");
        if state.step == 0 {
            Some(SetupAction::Bid {
                faction: Faction::AlphaStrike,
                bid: 0,
            })
        } else if state.can_place_factory().unwrap()
            && state.my_team().unwrap().factories_to_place > 0
        {
            let valid_spawns = state.board.iter_valid_spawns().collect::<Vec<_>>();
            let idx: usize = self.rng.sample(Uniform::new(0, valid_spawns.len()));
            Some(SetupAction::Spawn {
                spawn: valid_spawns[idx],
                metal: 150,
                water: 150,
            })
        } else {
            None
        }
    }
    fn act(&mut self, state: &State) -> UnitActions {
        debug!("requested act");
        let obstruction_board = build_obstruction_board(state);
        let closest_ice_map = build_closest_ice_map(state, Some(&obstruction_board));
        let closest_self_factory_map =
            build_closest_self_factory_map(state, Some(&obstruction_board));
        state
            .my_factories()
            .iter()
            .map(|(id, factory)| {
                if ((factory.water_cost(state) + 200) as f64) < (factory.cargo.water as f64 / 5.) {
                    Some(UnitAction::factory_water())
                } else if factory.can_build_heavy(&state.env_cfg) {
                    Some(UnitAction::factory_build_heavy())
                } else {
                    None
                }
                .map(|x| (id.clone(), x))
            })
            .filter(Option::is_some)
            .flatten()
            .chain(
                state
                    .my_units()
                    .iter()
                    .map(|(id, robot)| {
                        gather_ice_logic(&closest_ice_map, &closest_self_factory_map, robot, state)
                            .map(|x| (id.clone(), x))
                    })
                    .filter(Option::is_some)
                    .flatten(),
            )
            .collect()
    }
}
