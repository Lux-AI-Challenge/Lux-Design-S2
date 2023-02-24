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

#[derive(Debug, Clone)]
pub enum ResourceType {
    Ice = 0,
    Ore = 1,
    Water = 2,
    Metal = 3,
    Power = 4,
}

#[derive(Debug, Clone)]
pub struct UnitActionCommand {
    action: UnitAction,
    repeat: u64,
    n: u64,
}

#[derive(Debug, Clone)]
pub enum UnitAction {
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

pub type UnitActions = std::collections::HashMap<String, UnitActionCommand>;

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SetupAction {
    Spawn {
        spawn: Vec<u64>,
        metal: u64,
        water: u64,
    },
    Bid {
        faction: String,
        // TODO(seamooo) is u64 enough for this?
        bid: u64,
    },
}

impl Serialize for UnitActionCommand {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let (ty, direction, resource_type, amount): (u64, u64, u64, u64) = match &self.action {
            UnitAction::Move { direction } => (
                0,
                direction.clone() as u64,
                Default::default(),
                Default::default(),
            ),
            UnitAction::Transfer {
                direction,
                resource_type,
                amount,
            } => (
                1,
                direction.clone() as u64,
                resource_type.clone() as u64,
                *amount,
            ),
            UnitAction::Pickup {
                resource_type,
                amount,
            } => (2, Default::default(), resource_type.clone() as u64, *amount),
            UnitAction::Dig => (
                3,
                Default::default(),
                Default::default(),
                Default::default(),
            ),
            UnitAction::SelfDestruct => (
                4,
                Default::default(),
                Default::default(),
                Default::default(),
            ),
            UnitAction::Recharge => (
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

impl<'de> Deserialize<'de> for UnitActionCommand {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let array = <[u64; 6] as Deserialize<'de>>::deserialize(deserializer)?;
        let ty = array[0];
        let direction = array[1];
        let resource_type = array[2];
        let amount = array[3];
        let repeat = array[4];
        let n = array[5];
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
            0 => Ok(UnitAction::Move {
                direction: make_direction()?,
            }),
            1 => Ok(UnitAction::Transfer {
                direction: make_direction()?,
                resource_type: make_resource_type()?,
                amount,
            }),
            2 => Ok(UnitAction::Pickup {
                resource_type: make_resource_type()?,
                amount,
            }),
            3 => Ok(UnitAction::Dig),
            4 => Ok(UnitAction::SelfDestruct),
            5 => Ok(UnitAction::Recharge),
            _ => Err(D::Error::invalid_value(
                de::Unexpected::Unsigned(direction),
                &"expected a value in the range 0..=5",
            )),
        }?;
        // TODO(seamooo) should validation for limits on repeat / n be done here
        Ok(Self { action, repeat, n })
    }
}
