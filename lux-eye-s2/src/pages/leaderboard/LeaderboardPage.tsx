import { Center, Container, Grid, Loader, Paper, SimpleGrid, Space, Stack, Text, Title } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStore } from '../../store';
import { notifyError } from '../../utils/notifications';
import { Board } from '../visualizer/Board';
import { TeamBanner } from '../visualizer/TeamBanner';
import { TeamCard } from '../visualizer/TeamCard';
import { TurnControl } from '../visualizer/TurnControl';

let hasData = false;

export function LeaderboardPage(): JSX.Element {
  const episode = useStore(state => state.episode);
  const load = useStore(state => state.load);

  const { ref: boardContainerRef, width: maxBoardWidth } = useElementSize();

  const [seconds, setSeconds] = useState(0);
  const navigate = useNavigate();

  const onWindowMessage = useCallback((event: MessageEvent<any>) => {
    if (hasData) {
      return;
    }

    if (event.data && event.data.environment) {
      hasData = true;

      try {
        load(event.data.environment);
      } catch (err: any) {
        console.error(err);
        notifyError('Cannot load episode from Kaggle', err.message);
        navigate('/');
      }
    }
  }, []);

  useEffect(() => {
    window.addEventListener('message', onWindowMessage);
    return () => {
      window.removeEventListener('message', onWindowMessage);
    };
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setSeconds(s => s + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  if (episode === null) {
    return (
      <Center style={{ height: '100vh' }}>
        <div style={{ textAlign: 'center' }}>
          <Loader />
          <Title>Waiting for episode data</Title>
          {seconds >= 3 && <Text>This is taking longer than expected, please reopen the replay.</Text>}
        </div>
      </Center>
    );
  }

  const tabHeight = 350;

  return (
    <Container fluid>
      <Grid columns={24}>
        <Grid.Col span={24} xs={18} offsetXs={3}>
          <Paper shadow="xs" p="xs" withBorder>
            <Stack spacing={4}>
              <SimpleGrid cols={2}>
                <TeamBanner id={0} alignLeft={true} />
                <TeamBanner id={1} alignLeft={false} />
              </SimpleGrid>
              <Center ref={boardContainerRef}>
                <Board maxWidth={maxBoardWidth} />
              </Center>
              <Space h={4} />
              <TurnControl showHotkeysButton={false} showOpenButton={true} />
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
