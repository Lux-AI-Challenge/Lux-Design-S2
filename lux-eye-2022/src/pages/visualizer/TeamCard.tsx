import { Badge, Grid, Paper, Space, Tabs, Title } from '@mantine/core';
import { IconCrown } from '@tabler/icons';
import { useCallback, useMemo } from 'react';
import { Episode, Faction, RobotType, SetupAction, Unit } from '../../episode/model';
import { getFactoryTiles } from '../../episode/utils';
import { useStore } from '../../store';
import { getTeamColor } from '../../utils/colors';
import { FactoryDetail } from './FactoryDetail';
import { RobotDetail } from './RobotDetail';
import { UnitList } from './UnitList';

function getWinnerInfo(episode: Episode, team: number): [won: boolean, reason: string | null] {
  const lastStep = episode.steps[episode.steps.length - 1];

  const me = lastStep.teams[team];
  const opponent = lastStep.teams[team === 0 ? 1 : 0];

  const meLichen = me.factories.map(factory => factory.lichen).reduce((acc, val) => acc + val, 0);
  const opponentLichen = opponent.factories.map(factory => factory.lichen).reduce((acc, val) => acc + val, 0);

  if (me.faction !== Faction.None && opponent.faction === Faction.None) {
    return [true, 'Winner by opponent error'];
  } else if (me.faction === Faction.None && opponent.faction !== Faction.None) {
    return [false, null];
  } else if (me.factories.length === 0 && opponent.factories.length === 0) {
    return [true, 'Draw, all factories destroyed'];
  } else if (me.factories.length > 0 && opponent.factories.length === 0) {
    return [true, 'Winner by factory elimination'];
  } else if (me.factories.length === 0 && opponent.factories.length > 0) {
    return [false, null];
  } else if (lastStep.step === 1000) {
    if (meLichen > opponentLichen) {
      return [true, 'Winner by lichen'];
    } else if (meLichen === opponentLichen) {
      return [true, 'Draw, same lichen'];
    } else {
      return [false, null];
    }
  } else {
    return [true, 'Draw, game ended prematurely'];
  }
}

function compareUnits(a: Unit, b: Unit): number {
  const partsA = a.unitId.split('_');
  const partsB = b.unitId.split('_');

  if (partsA[0] === partsB[0]) {
    return parseInt(partsA[1]) - parseInt(partsB[1]);
  }

  return partsA[0].localeCompare(partsB[0]);
}

function formatFaction(faction: Faction): string {
  switch (faction) {
    case Faction.None:
      return 'No Faction';
    case Faction.AlphaStrike:
      return 'Alpha Strike';
    case Faction.MotherMars:
      return 'Mother Mars';
    case Faction.TheBuilders:
      return 'The Builders';
    case Faction.FirstMars:
      return 'First Mars';
  }
}

function formatAction(action: SetupAction): string {
  switch (action.type) {
    case 'bid':
      return `Bid ${action.bid} and choose ${formatFaction(action.faction)} faction`;
    case 'buildFactory':
      return `Build factory on (${action.center.x}, ${action.center.y}) with ${action.water} water and ${action.metal} metal`;
    case 'wait':
      return 'None';
  }
}

interface TeamCardProps {
  id: number;
  tabHeight: number;
}

export function TeamCard({ id, tabHeight }: TeamCardProps): JSX.Element {
  const episode = useStore(state => state.episode)!;
  const turn = useStore(state => state.turn);

  const step = episode.steps[turn];
  const team = step.teams[id];

  const [isWinner, winnerReason] = getWinnerInfo(episode, id);

  const sortedFactories = useMemo(() => team.factories.sort(compareUnits), [team]);
  const factoryRenderer = useCallback(
    (index: number) => <FactoryDetail factory={sortedFactories[index]} />,
    [sortedFactories],
  );
  const factoryTileGetter = useCallback((index: number) => getFactoryTiles(sortedFactories[index]), [sortedFactories]);

  const sortedRobots = useMemo(() => team.robots.sort(compareUnits), [team]);
  const robotRenderer = useCallback((index: number) => <RobotDetail robot={sortedRobots[index]} />, [sortedRobots]);
  const robotTileGetter = useCallback((index: number) => [sortedRobots[index].tile], [sortedRobots]);

  tabHeight = step.step < 0 ? tabHeight - 100 : tabHeight;
  tabHeight = isWinner ? tabHeight - 24 : tabHeight;

  return (
    <Paper shadow="xs" p="md" withBorder>
      <Title order={3} style={{ color: getTeamColor(id, 1.0) }}>
        {isWinner && <IconCrown style={{ verticalAlign: 'middle', marginRight: '2px' }} />}
        {team.name}
      </Title>

      {isWinner && <Badge color={id === 0 ? 'blue' : 'red'}>{winnerReason}</Badge>}

      <Space h="xs" />

      <Grid columns={2} gutter={0}>
        <Grid.Col span={1}>
          <b>Lichen:</b> {team.factories.map(factory => factory.lichen).reduce((acc, val) => acc + val, 0)}
        </Grid.Col>
        <Grid.Col span={1}>
          <b>Light robots:</b> {sortedRobots.filter(robot => robot.type === RobotType.Light).length}
        </Grid.Col>
        <Grid.Col span={1}>
          <b>Factories:</b> {sortedFactories.length}
        </Grid.Col>
        <Grid.Col span={1}>
          <b>Heavy robots:</b> {sortedRobots.filter(robot => robot.type === RobotType.Heavy).length}
        </Grid.Col>

        {step.step < 0 && (
          <>
            <Grid.Col span={1}>
              <b>Water:</b> {team.water}
            </Grid.Col>
            <Grid.Col span={1}>
              <b>Metal:</b> {team.metal}
            </Grid.Col>
            <Grid.Col span={1}>
              <b>Place first:</b> {team.placeFirst ? 'Yes' : 'No'}
            </Grid.Col>
            <Grid.Col span={1}>
              <b>Factories left:</b> {team.factoriesToPlace}
            </Grid.Col>
            <Grid.Col span={2}>
              <b>Action:</b> {team.action !== null ? formatAction(team.action) : 'None'}
            </Grid.Col>
          </>
        )}
      </Grid>

      <Space h="xs" />

      <Tabs defaultValue="factories" keepMounted={false} color={id === 0 ? 'blue' : 'red'}>
        <Tabs.List mb="xs" grow>
          <Tabs.Tab value="factories">Factories</Tabs.Tab>
          <Tabs.Tab value="robots">Robots</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="factories">
          <UnitList
            name="factories"
            height={tabHeight}
            itemCount={sortedFactories.length}
            itemRenderer={factoryRenderer}
            tileGetter={factoryTileGetter}
          />
        </Tabs.Panel>
        <Tabs.Panel value="robots">
          <UnitList
            name="robots"
            height={tabHeight}
            itemCount={sortedRobots.length}
            itemRenderer={robotRenderer}
            tileGetter={robotTileGetter}
          />
        </Tabs.Panel>
      </Tabs>
    </Paper>
  );
}
