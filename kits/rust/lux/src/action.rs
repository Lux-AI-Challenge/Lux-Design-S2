//! Handles unit and setup actions. Additionally enables serializing,
//! and deserializing of aforementioned actions.

use crate::team::Faction;
use crate::Pos;
use serde::{
    de::{self, Error as DeError},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_repr::{Deserialize_repr, Serialize_repr};

/// Represents direction for move and transfer actions
///
/// [`Direction::Center`] represents a no-op for move
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub enum Direction {
    Center = 0,
    Up = 1,
    Right = 2,
    Down = 3,
    Left = 4,
}

impl Direction {
    /// Get position delta represented by the direction
    #[inline(always)]
    pub fn to_pos(&self) -> Pos {
        match self {
            Self::Center => Pos(0, 0),
            Self::Up => Pos(0, -1),
            Self::Right => Pos(1, 0),
            Self::Down => Pos(0, 1),
            Self::Left => Pos(-1, 0),
        }
    }

    /// Iterate across all directions
    #[inline(always)]
    pub fn iter_all() -> impl Iterator<Item = Self> {
        [Self::Center, Self::Up, Self::Right, Self::Down, Self::Left].into_iter()
    }

    /// Calculate the next move direction for a given source, destination pair
    ///
    /// # Note
    ///
    /// This does not account for obstructions and as such should not be trusted
    /// to always path towards the destination
    pub fn move_towards(src: &Pos, dst: &Pos) -> Self {
        let dx = dst.0 - src.0;
        let dy = dst.1 - src.1;
        if dx == dy && dx == 0 {
            Direction::Center
        } else {
            match dx.abs().cmp(&dy.abs()) {
                std::cmp::Ordering::Greater => match dx.signum() {
                    1 => Direction::Right,
                    _ => Direction::Left,
                },
                _ => match dy.signum() {
                    1 => Direction::Down,
                    _ => Direction::Up,
                },
            }
        }
    }
}

/// Type of resource used in action
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub enum ResourceType {
    Ice = 0,
    Ore = 1,
    Water = 2,
    Metal = 3,
    Power = 4,
}

/// Higher level representation of a unit action vector
#[derive(Debug, Clone)]
pub struct RobotActionCommand {
    /// The action to perform
    pub action: RobotAction,

    /// Should this action be recycled by the [`action_queue`](crate::robot::Robot::action_queue)
    pub repeat: bool,

    /// How many times the action should be executed before cycling the action queue
    pub n: u64,
}

/// Action a robot can perform.
///
/// Should only be used as part of constructing a [`RobotActionCommand`]
#[derive(Debug, Clone)]
pub enum RobotAction {
    /// Moves robot in the direction given.
    ///
    /// Moving to [`Direction::Center`] is a no-op
    Move {
        /// Direction of movement
        direction: Direction,
    },
    /// Transfers an amount of a resource to another unit in the given direction
    ///
    /// # Notes
    /// - The receiving unit can only receive up to it's capacity
    /// - Overflow resources are wasted
    Transfer {
        /// Direction of transfer
        direction: Direction,
        /// Type of resource to transfer
        resource_type: ResourceType,
        /// Amount of resource to transfer
        amount: u64,
    },
    /// Pickup an amount of resource on the factory present at the same tile
    /// as the robot
    ///
    /// # Notes
    /// - Preference is given to robots with lower ids
    /// - Pickup not on a factory is invalid
    Pickup {
        /// Type of resource to pickup
        resource_type: ResourceType,
        /// Amount of resource to pickup
        amount: u64,
    },
    /// Performs a dig at the same tile as the unit
    ///
    /// If the tile is a resource tile (ore / ice) it retrieves the resource.
    ///
    /// If the tile is rubble the rubble is reduced by 2 if a light robot is performing
    /// the action and 20 if a heavy robot is instead.
    ///
    /// If the tile is lichen the lichen value is reduced by 10. If this reduces this value
    /// to zero, rubble is added. 20 rubble is added if the unit is heavy, and 2 if it is instead
    /// light.
    Dig,
    /// Destroys the robot on the spot, creating rubble corresponding to
    /// [`RobotTypeConfig::rubble_after_desctruction`](crate::config::RobotTypeConfig::dig_cost)
    SelfDestruct,
    /// Waits until robot has `amount` power
    ///
    /// This action is not removed from the queue until the robot has `amount` power
    Recharge {
        /// Target power to wait until
        amount: u64,
    },
}

impl RobotActionCommand {
    #[inline(always)]
    fn default_repeat_n(repeat_n: (Option<bool>, Option<u64>)) -> (bool, u64) {
        let (repeat, n) = repeat_n;
        (repeat.unwrap_or(false), n.unwrap_or(1))
    }
    /// Constructs a [`RobotAction::Move`] action as an action command
    pub fn move_(direction: Direction, repeat: Option<bool>, n: Option<u64>) -> Self {
        let (repeat, n) = Self::default_repeat_n((repeat, n));
        let action = RobotAction::Move { direction };
        Self { action, repeat, n }
    }
    /// Constructs a [`RobotAction::Transfer`] action as an action command
    pub fn transfer(
        direction: Direction,
        resource_type: ResourceType,
        amount: u64,
        repeat: Option<bool>,
        n: Option<u64>,
    ) -> Self {
        let (repeat, n) = Self::default_repeat_n((repeat, n));
        let action = RobotAction::Transfer {
            direction,
            resource_type,
            amount,
        };
        Self { action, repeat, n }
    }
    /// Constructs a [`RobotAction::Pickup`] action as an action command
    pub fn pickup(
        resource_type: ResourceType,
        amount: u64,
        repeat: Option<bool>,
        n: Option<u64>,
    ) -> Self {
        let (repeat, n) = Self::default_repeat_n((repeat, n));
        let action = RobotAction::Pickup {
            resource_type,
            amount,
        };
        Self { action, repeat, n }
    }
    /// Constructs a [`RobotAction::Dig`] action as an action command
    pub fn dig(repeat: Option<bool>, n: Option<u64>) -> Self {
        let (repeat, n) = Self::default_repeat_n((repeat, n));
        Self {
            action: RobotAction::Dig,
            repeat,
            n,
        }
    }
    /// Constructs a [`RobotAction::SelfDestruct`] action as an action command
    pub fn self_destruct() -> Self {
        Self {
            action: RobotAction::SelfDestruct,
            repeat: false,
            n: 1,
        }
    }
    /// Constructs a [`RobotAction::Recharge`] action as an action command
    pub fn recharge(amount: u64, repeat: Option<bool>, n: Option<u64>) -> Self {
        let (repeat, n) = Self::default_repeat_n((repeat, n));
        Self {
            action: RobotAction::Recharge { amount },
            repeat,
            n,
        }
    }
    /// Constructs a [`RobotAction::Move`] action as an action command
    /// in the direction given by `src` to `dest`
    pub fn move_towards(src: &Pos, dst: &Pos) -> Self {
        let direction = Direction::move_towards(src, dst);
        let action = RobotAction::Move { direction };
        Self {
            action,
            repeat: false,
            n: 1,
        }
    }
}

/// An action able to be performed by a factory unit
#[derive(Serialize_repr, Deserialize_repr, Debug)]
#[repr(u8)]
pub enum FactoryAction {
    /// Build a light robot
    BuildLight = 0,

    /// Build a heavy robot
    BuildHeavy = 1,

    /// Grow lichen by watering lichen around the factory
    Water = 2,
}

/// A type encapsulating both robot and factory actions for serialization,
/// deserialization and storage.
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum UnitAction {
    /// A vector of robot actions to be performed for a single unit
    Robot(Vec<RobotActionCommand>),

    /// A factory action performed by a single unit
    Factory(FactoryAction),
}

impl UnitAction {
    /// Constructs a [`UnitAction`] from [`FactoryAction::BuildLight`]
    #[inline(always)]
    pub fn factory_build_light() -> Self {
        Self::Factory(FactoryAction::BuildLight)
    }

    /// Constructs a [`UnitAction`] from [`FactoryAction::BuildHeavy`]
    #[inline(always)]
    pub fn factory_build_heavy() -> Self {
        Self::Factory(FactoryAction::BuildHeavy)
    }

    /// Constructs a [`UnitAction`] from [`FactoryAction::Water`]
    #[inline(always)]
    pub fn factory_water() -> Self {
        Self::Factory(FactoryAction::Water)
    }
}

/// Unit id to [`UnitAction`] mapping expected to be returned from an
/// agent in the action phase
pub type UnitActions = std::collections::HashMap<String, UnitAction>;

/// Actions possible to perform in the setup phase
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SetupAction {
    /// Spawns a factory, and supplies it with the given amonut of metal and water
    Spawn {
        /// Center of the 3x3 factory
        spawn: Pos,
        /// Starting supply of metal for the factory
        metal: u64,
        /// Starting supply of water for the factory
        water: u64,
    },
    /// Selects faction places a bid on the maximum number of resources offered for
    /// the chance to go first.
    Bid {
        /// Faction to initialize as.
        ///
        /// Useful for coloring your agent in a gui
        faction: Faction,

        /// Bid for the maximum resources offered for the chance to go first.
        ///
        /// If the bid is successful, this many water, and metal resources will be removed
        bid: u64,
    },
}

impl Serialize for RobotActionCommand {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let (ty, direction, resource_type, amount): (u64, u64, u64, u64) = match &self.action {
            RobotAction::Move { direction } => (
                0,
                direction.clone() as u64,
                Default::default(),
                Default::default(),
            ),
            RobotAction::Transfer {
                direction,
                resource_type,
                amount,
            } => (
                1,
                direction.clone() as u64,
                resource_type.clone() as u64,
                *amount,
            ),
            RobotAction::Pickup {
                resource_type,
                amount,
            } => (2, Default::default(), resource_type.clone() as u64, *amount),
            RobotAction::Dig => (
                3,
                Default::default(),
                Default::default(),
                Default::default(),
            ),
            RobotAction::SelfDestruct => (
                4,
                Default::default(),
                Default::default(),
                Default::default(),
            ),
            RobotAction::Recharge { amount } => {
                (5, Default::default(), Default::default(), *amount)
            }
        };
        let repeat: u64 = if self.repeat { 1 } else { 0 };
        let mut rv = serializer.serialize_tuple(6)?;
        rv.serialize_element(&ty)?;
        rv.serialize_element(&direction)?;
        rv.serialize_element(&resource_type)?;
        rv.serialize_element(&amount)?;
        rv.serialize_element(&repeat)?;
        rv.serialize_element(&self.n)?;
        rv.end()
    }
}

impl<'de> Deserialize<'de> for RobotActionCommand {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (ty, direction, resource_type, amount, repeat_u64, n) =
            <(u64, u64, u64, u64, u64, u64) as Deserialize<'de>>::deserialize(deserializer)?;
        // below technically allows out of bounds values for repeat but will behave
        // correctly for any server implementing the spec correctly
        let repeat = repeat_u64 > 0;
        // lazily do enum conversions as they are don't cares prior
        let make_direction = || match direction {
            0 => Ok(Direction::Center),
            1 => Ok(Direction::Up),
            2 => Ok(Direction::Right),
            3 => Ok(Direction::Down),
            4 => Ok(Direction::Left),
            _ => Err(D::Error::invalid_value(
                de::Unexpected::Unsigned(direction),
                &"expected a value in the range 0..=4",
            )),
        };
        let make_resource_type = || match resource_type {
            0 => Ok(ResourceType::Ice),
            1 => Ok(ResourceType::Ore),
            2 => Ok(ResourceType::Water),
            3 => Ok(ResourceType::Metal),
            4 => Ok(ResourceType::Power),
            _ => Err(D::Error::invalid_value(
                de::Unexpected::Unsigned(direction),
                &"expected a value in the range 0..=4",
            )),
        };
        let action = match ty {
            0 => Ok(RobotAction::Move {
                direction: make_direction()?,
            }),
            1 => Ok(RobotAction::Transfer {
                direction: make_direction()?,
                resource_type: make_resource_type()?,
                amount,
            }),
            2 => Ok(RobotAction::Pickup {
                resource_type: make_resource_type()?,
                amount,
            }),
            3 => Ok(RobotAction::Dig),
            4 => Ok(RobotAction::SelfDestruct),
            5 => Ok(RobotAction::Recharge { amount }),
            _ => Err(D::Error::invalid_value(
                de::Unexpected::Unsigned(direction),
                &"expected a value in the range 0..=5",
            )),
        }?;
        if n == 0 || n > 9999 {
            return Err(de::Error::invalid_value(
                de::Unexpected::Unsigned(direction),
                &"n must be between 1 and 9999 inclusive",
            ));
        }
        Ok(Self { action, repeat, n })
    }
}
