import { Grid, Text } from '@mantine/core';
import { Factory, FactoryAction, RobotType } from '../../episode/model';
import { getFactoryTiles } from '../../episode/utils';
import { UnitCard } from './UnitCard';

function formatAction(action: FactoryAction | null): string {
  if (action === null) {
    return 'None';
  }

  switch (action.type) {
    case 'buildRobot':
      return `Build ${RobotType[action.robotType]}`;
    case 'water':
      return 'Water';
  }
}

interface FactoryDetailProps {
  factory: Factory;
}

export function FactoryDetail({ factory }: FactoryDetailProps): JSX.Element {
  return (
    <UnitCard tiles={getFactoryTiles(factory)} tileToSelect={factory.tile}>
      <Grid gutter={0}>
        <Grid.Col span={6}>
          <Text size="sm">
            <b>{factory.unitId}</b>
          </Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">
            Center: ({factory.tile.x}, {factory.tile.y})
          </Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Strain: {factory.strain}</Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Lichen: {factory.lichen}</Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Ice: {factory.cargo.ice}</Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Water: {factory.cargo.water}</Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Ore: {factory.cargo.ore}</Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Metal: {factory.cargo.metal}</Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Power: {factory.power}</Text>
        </Grid.Col>
        <Grid.Col span={6}>
          <Text size="sm">Action: {formatAction(factory.action)}</Text>
        </Grid.Col>
      </Grid>
    </UnitCard>
  );
}
