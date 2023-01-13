import { Grid, Indicator, Text, Tooltip } from '@mantine/core';
import {
  IconArrowDown,
  IconArrowLeft,
  IconArrowRight,
  IconArrowsTransferDown,
  IconArrowUp,
  IconBatteryCharging,
  IconBomb,
  IconHandGrab,
  IconShovel,
  IconWalk,
  TablerIconProps,
} from '@tabler/icons';
import { Direction, Resource, Robot, RobotAction, RobotType } from '../../episode/model';
import { UnitCard } from './UnitCard';

interface RobotDetailProps {
  robot: Robot;
}

function formatAction(action: RobotAction): string {
  switch (action.type) {
    case 'move': {
      const direction = Direction[action.direction].toLowerCase();
      return `Move ${direction}`;
    }
    case 'transfer': {
      const direction = Direction[action.direction].toLowerCase();
      const resource = Resource[action.resource].toLowerCase();
      return `Transfer ${direction} ${action.amount} ${resource}`;
    }
    case 'pickup': {
      const resource = Resource[action.resource].toLowerCase();
      return `Pickup ${action.amount} ${resource}`;
    }
    case 'dig': {
      return 'Dig';
    }
    case 'selfDestruct': {
      return 'Self destruct';
    }
    case 'recharge': {
      return `Recharge to ${action.targetPower}`;
    }
  }
}

function getActionIcon(action: RobotAction): JSX.Element {
  const props: TablerIconProps = {
    size: 20,
    style: {
      verticalAlign: 'middle',
    },
  };

  switch (action.type) {
    case 'move':
      switch (action.direction) {
        case Direction.Center:
          return <IconWalk {...props} />;
        case Direction.Up:
          return <IconArrowUp {...props} />;
        case Direction.Right:
          return <IconArrowRight {...props} />;
        case Direction.Down:
          return <IconArrowDown {...props} />;
        case Direction.Left:
          return <IconArrowLeft {...props} />;
      }
      break;
    case 'transfer':
      return <IconArrowsTransferDown {...props} />;
    case 'pickup':
      return <IconHandGrab {...props} />;
    case 'dig':
      return <IconShovel {...props} />;
    case 'selfDestruct':
      return <IconBomb {...props} />;
    case 'recharge':
      return <IconBatteryCharging {...props} />;
  }
}

export function RobotDetail({ robot }: RobotDetailProps): JSX.Element {
  const actionQueueIcons = [];
  for (const action of robot.actionQueue) {
    let icon: JSX.Element;
    let suffix: string;

    if (action.repeat != 0) {
      icon = (
        <Indicator inline color="dark" mt={4} size={12} label={action.repeat.toString()}>
          {getActionIcon(action)}
        </Indicator>
      );

      suffix = ` (repeat: ${action.repeat})`;
    } else {
      icon = <span>{getActionIcon(action)}</span>;
      suffix = '';
    }

    actionQueueIcons.push(
      <Tooltip key={actionQueueIcons.length} label={formatAction(action) + suffix}>
        {icon}
      </Tooltip>,
    );
  }

  return (
    <UnitCard tiles={[robot.tile]} tileToSelect={robot.tile}>
      <Grid gutter={0}>
        <Grid.Col span={6}>
          <Text size="sm">
            <b>{robot.unitId}</b>
          </Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">
            Tile: ({robot.tile.x}, {robot.tile.y})
          </Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Power: {robot.power}</Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Type: {RobotType[robot.type]}</Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Ice: {robot.cargo.ice}</Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Ore: {robot.cargo.ore}</Text>
        </Grid.Col>
        <Grid.Col span={12}>
          <Text size="sm">Action: {robot.actionQueue.length > 0 ? formatAction(robot.actionQueue[0]) : 'None'}</Text>
          {actionQueueIcons}
        </Grid.Col>
      </Grid>
    </UnitCard>
  );
}
