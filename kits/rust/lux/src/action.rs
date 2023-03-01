use crate::team::Faction;
use crate::Pos;
use serde::{
    de::{self, Error as DeError},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

#[derive(Debug, Clone)]
pub enum Direction {
    Center = 0,
    Up = 1,
    Right = 2,
    Down = 3,
    Left = 4,
}

impl Direction {
    #[inline(always)]
    pub fn to_pos(&self) -> Pos {
        match self {
            Self::Center => (0, 0),
            Self::Up => (0, -1),
            Self::Right => (1, 0),
            Self::Down => (0, 1),
            Self::Left => (-1, 0),
        }
    }
    #[inline(always)]
    pub fn iter_all() -> impl Iterator<Item = Self> {
        [Self::Center, Self::Up, Self::Right, Self::Down, Self::Left].into_iter()
    }
    pub fn move_towards(src: &Pos, dst: &Pos) -> Self {
        // TODO(seamooo) should have an optional obstruction map here
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

#[derive(Debug, Clone)]
pub enum ResourceType {
    Ice = 0,
    Ore = 1,
    Water = 2,
    Metal = 3,
    Power = 4,
}

#[derive(Debug, Clone)]
pub struct RobotActionCommand {
    pub action: RobotAction,
    pub repeat: u64,
    pub n: u64,
}

#[derive(Debug, Clone)]
pub enum RobotAction {
    Move {
        direction: Direction,
    },
    Transfer {
        direction: Direction,
        resource_type: ResourceType,
        amount: u64,
    },
    Pickup {
        resource_type: ResourceType,
        amount: u64,
    },
    Dig,
    SelfDestruct,
    Recharge,
}

impl RobotActionCommand {
    #[inline(always)]
    fn default_repeat_n(repeat_n: (Option<u64>, Option<u64>)) -> (u64, u64) {
        let (repeat, n) = repeat_n;
        (repeat.unwrap_or(0), n.unwrap_or(1))
    }
    pub fn move_(direction: Direction, repeat: Option<u64>, n: Option<u64>) -> Self {
        let (repeat, n) = Self::default_repeat_n((repeat, n));
        let action = RobotAction::Move { direction };
        Self { action, repeat, n }
    }
    pub fn transfer(
        direction: Direction,
        resource_type: ResourceType,
        amount: u64,
        repeat: Option<u64>,
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
    pub fn pickup(
        resource_type: ResourceType,
        amount: u64,
        repeat: Option<u64>,
        n: Option<u64>,
    ) -> Self {
        let (repeat, n) = Self::default_repeat_n((repeat, n));
        let action = RobotAction::Pickup {
            resource_type,
            amount,
        };
        Self { action, repeat, n }
    }
    pub fn dig(repeat: Option<u64>, n: Option<u64>) -> Self {
        let (repeat, n) = Self::default_repeat_n((repeat, n));
        Self {
            action: RobotAction::Dig,
            repeat,
            n,
        }
    }
    pub fn self_destruct(repeat: Option<u64>, n: Option<u64>) -> Self {
        // TODO(seamooo) should it be possible to repeat self destruct?
        let (repeat, n) = Self::default_repeat_n((repeat, n));
        Self {
            action: RobotAction::SelfDestruct,
            repeat,
            n,
        }
    }
    pub fn recharge(repeat: Option<u64>, n: Option<u64>) -> Self {
        let (repeat, n) = Self::default_repeat_n((repeat, n));
        Self {
            action: RobotAction::Recharge,
            repeat,
            n,
        }
    }
    pub fn move_towards(src: &Pos, dst: &Pos) -> Self {
        let direction = Direction::move_towards(src, dst);
        let action = RobotAction::Move { direction };
        Self {
            action,
            repeat: 0,
            n: 1,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub enum FactoryAction {
    BuildLight = 0,
    BuildHeavy = 1,
    Water = 2,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum UnitAction {
    Robot(Vec<RobotActionCommand>),
    Factory(FactoryAction),
}

impl UnitAction {
    #[inline(always)]
    pub fn factory_build_light() -> Self {
        Self::Factory(FactoryAction::BuildLight)
    }
    #[inline(always)]
    pub fn factory_build_heavy() -> Self {
        Self::Factory(FactoryAction::BuildHeavy)
    }
    #[inline(always)]
    pub fn factory_water() -> Self {
        Self::Factory(FactoryAction::Water)
    }
}

pub type UnitActions = std::collections::HashMap<String, UnitAction>;

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SetupAction {
    Spawn {
        spawn: Pos,
        metal: u64,
        water: u64,
    },
    Bid {
        faction: Faction,
        // TODO(seamooo) is u64 enough for this?
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
            RobotAction::Recharge => (
                5,
                Default::default(),
                Default::default(),
                Default::default(),
            ),
        };
        let mut rv = serializer.serialize_tuple(6)?;
        rv.serialize_element(&ty)?;
        rv.serialize_element(&direction)?;
        rv.serialize_element(&resource_type)?;
        rv.serialize_element(&amount)?;
        rv.serialize_element(&self.repeat)?;
        rv.serialize_element(&self.n)?;
        rv.end()
    }
}

impl<'de> Deserialize<'de> for RobotActionCommand {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (ty, direction, resource_type, amount, repeat, n) =
            <(u64, u64, u64, u64, u64, u64) as Deserialize<'de>>::deserialize(deserializer)?;
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
            5 => Ok(RobotAction::Recharge),
            _ => Err(D::Error::invalid_value(
                de::Unexpected::Unsigned(direction),
                &"expected a value in the range 0..=5",
            )),
        }?;
        // TODO(seamooo) should validation for limits on repeat / n be done here
        Ok(Self { action, repeat, n })
    }
}
