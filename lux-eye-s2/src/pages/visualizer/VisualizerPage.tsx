import { Center, createStyles, Grid, MediaQuery, Paper, Stack } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { Navigate, useLocation } from 'react-router-dom';
import { Cargo, RobotType, Unit } from '../../episode/model';
import { useStore } from '../../store';
import { Board } from './Board';
import { Chart, ChartFunction } from './Chart';
import { TeamCard } from './TeamCard';
import { TurnControl } from './TurnControl';

const useStyles = createStyles(theme => ({
  container: {
    margin: '0 auto',
    width: '1500px',

    [theme.fn.smallerThan(1500)]: {
      width: '100%',
    },
  },
}));

function funcCargo(unitType: 'factories' | 'robots', resource: keyof Cargo): ChartFunction {
  return team => (team[unitType] as Unit[]).reduce((acc, val) => acc + val.cargo[resource], 0);
}

export const funcLichen: ChartFunction = (team, board) => {
  let lichen = 0;

  const size = board.lichen.length;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (board.lichen[y][x] > 0 && team.strains.has(board.strains[y][x])) {
        lichen += board.lichen[y][x];
      }
    }
  }

  return lichen;
};

const funcTotalMetalValue: ChartFunction = team =>
  team.factories.reduce((acc, val) => acc + val.cargo.metal, 0) +
  team.robots.filter(robot => robot.type === RobotType.Light).length * 10 +
  team.robots.filter(robot => robot.type === RobotType.Heavy).length * 100 +
  0.2 * team.robots.reduce((acc, val) => acc + val.cargo.ore, 0) +
  0.2 * team.factories.reduce((acc, val) => acc + val.cargo.ore, 0);

const funcFactories: ChartFunction = team => team.factories.length;
const funcFactoryPower: ChartFunction = team => team.factories.reduce((acc, val) => acc + val.power, 0);

const funcRobots: ChartFunction = team => team.robots.length;
const funcLightRobots: ChartFunction = team => team.robots.filter(robot => robot.type === RobotType.Light).length;
const funcHeavyRobots: ChartFunction = team => team.robots.filter(robot => robot.type === RobotType.Heavy).length;
const funcRobotPower: ChartFunction = team => team.robots.reduce((acc, val) => acc + val.power, 0);

export function VisualizerPage(): JSX.Element {
  const { classes } = useStyles();

  const episode = useStore(state => state.episode);

  const { search } = useLocation();

  const { ref: boardContainerRef, width: maxBoardWidth } = useElementSize();

  if (episode === null) {
    return <Navigate to={`/${search}`} />;
  }

  const teamCards = [];
  for (let i = 0; i < 2; i++) {
    teamCards.push(<TeamCard id={i} tabHeight={570} shadow="xs" />);
  }

  return (
    <div className={classes.container}>
      <Grid columns={24}>
        <MediaQuery smallerThan="md" styles={{ display: 'none' }}>
          <Grid.Col span={7}>{teamCards[0]}</Grid.Col>
        </MediaQuery>
        <Grid.Col span={24} md={10}>
          <Paper shadow="xs" p="md" withBorder>
            <Stack>
              <Center ref={boardContainerRef}>
                <Board maxWidth={maxBoardWidth} />
              </Center>
              <TurnControl showHotkeysButton={true} showOpenButton={false} />
            </Stack>
          </Paper>
        </Grid.Col>
        <MediaQuery largerThan="md" styles={{ display: 'none' }}>
          <Grid.Col span={24}>{teamCards[0]}</Grid.Col>
        </MediaQuery>
        <Grid.Col span={24} md={7}>
          {teamCards[1]}
        </Grid.Col>
      </Grid>
      <Grid columns={12}>
        <Grid.Col span={12} md={6}>
          <Chart title="Lichen" func={funcLichen} />
        </Grid.Col>
        <Grid.Col span={12} md={6}>
          <Chart title="Power in robots" func={funcRobotPower} />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Total metal value" func={funcTotalMetalValue} />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Ice in robots" func={funcCargo('robots', 'ice')} />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Ore in robots" func={funcCargo('robots', 'ore')} />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Power in factories" func={funcFactoryPower} />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Ice in factories" func={funcCargo('factories', 'ice')} />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Water in factories" func={funcCargo('factories', 'water')} />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Factories" func={funcFactories} step />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Ore in factories" func={funcCargo('factories', 'ore')} />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Metal in factories" func={funcCargo('factories', 'metal')} />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Robots" func={funcRobots} step />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Light robots" func={funcLightRobots} step />
        </Grid.Col>
        <Grid.Col span={12} md={4}>
          <Chart title="Heavy robots" func={funcHeavyRobots} step />
        </Grid.Col>
      </Grid>
    </div>
  );
}
