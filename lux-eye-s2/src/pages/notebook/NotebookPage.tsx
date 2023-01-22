import { Center, Container, Grid, Loader, Paper, Stack, Text, Title } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStore } from '../../store';
import { notifyError } from '../../utils/notifications';
import { Board } from '../visualizer/Board';
import { TeamCard } from '../visualizer/TeamCard';
import { TurnControl } from '../visualizer/TurnControl';

let hasData = false;

export function NotebookPage(): JSX.Element {
  const episode = useStore(state => state.episode);
  const load = useStore(state => state.load);

  const { ref: boardContainerRef, width: maxBoardWidth } = useElementSize();

  const [seconds, setSeconds] = useState(0);
  const navigate = useNavigate();

  const onWindowMessage = useCallback((event: MessageEvent<any>) => {
    if (hasData) {
      return;
    }

    if (event.data && event.data.observations && event.data.actions) {
      hasData = true;

      try {
        load(event.data);
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
          <Title>Waiting for episode data from Kaggle</Title>
          {seconds >= 3 && <Text>This is taking longer than expected, please rerun the cell or refresh the page.</Text>}
        </div>
      </Center>
    );
  }

  const tabHeight = 346;

  return (
    <Container fluid pl={4} pr={4}>
      <Grid columns={24} gutter={8}>
        <Grid.Col span={7} pl={0}>
          <TeamCard id={0} tabHeight={tabHeight} />
        </Grid.Col>
        <Grid.Col span={10}>
          <Paper p="xs" withBorder>
            <Stack>
              <Center ref={boardContainerRef}>
                <Board maxWidth={maxBoardWidth} />
              </Center>
              <TurnControl showHotkeysButton={false} showOpenButton={true} />
            </Stack>
          </Paper>
        </Grid.Col>
        <Grid.Col span={7} pr={0}>
          <TeamCard id={1} tabHeight={tabHeight} />
        </Grid.Col>
      </Grid>
    </Container>
  );
}
