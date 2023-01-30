import { Button, Center, Container, Grid, Paper, SimpleGrid, Space, Stack } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { Navigate } from 'react-router-dom';
import { useStore } from '../../store';
import { Board } from '../visualizer/Board';
import { TeamBanner } from '../visualizer/TeamBanner';
import { TeamCard } from '../visualizer/TeamCard';
import { TurnControl } from '../visualizer/TurnControl';

export function LeaderboardPage(): JSX.Element {
  const episode = useStore(state => state.episode);
  const openInNewTab = useStore(state => state.openInNewTab);

  const { ref: boardContainerRef, width: maxBoardWidth } = useElementSize();

  if (episode === null) {
    return <Navigate to="/" />;
  }

  const tabHeight = 350;

  return (
    <Container fluid>
      <Grid columns={24}>
        <Grid.Col span={24} xs={18} offsetXs={3}>
          <Paper shadow="xs" p="xs" withBorder>
            <Stack spacing={4}>
              <SimpleGrid cols={3}>
                <TeamBanner id={0} alignLeft={true} />
                <Button compact color="blue" variant="subtle" onClick={openInNewTab}>
                  Open in full visualizer
                </Button>
                <TeamBanner id={1} alignLeft={false} />
              </SimpleGrid>
              <Center ref={boardContainerRef}>
                <Board maxWidth={maxBoardWidth} />
              </Center>
              <Space h={4} />
              <TurnControl showHotkeysButton={true} showOpenButton={false} />
            </Stack>
          </Paper>
        </Grid.Col>
        <Grid.Col span={24} xs={12}>
          <TeamCard id={0} tabHeight={tabHeight} />
        </Grid.Col>
        <Grid.Col span={24} xs={12}>
          <TeamCard id={1} tabHeight={tabHeight} />
        </Grid.Col>
      </Grid>
    </Container>
  );
}
