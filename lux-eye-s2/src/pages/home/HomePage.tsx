import { Container, Stack, Text } from '@mantine/core';
import { useCallback, useEffect } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { useStore } from '../../store';
import { notifyError } from '../../utils/notifications';
import { HomeCard } from './HomeCard';
import { LoadFromElsewhere } from './LoadFromElsewhere';
import { LoadFromFile } from './LoadFromFile';
import { LoadFromKaggle } from './LoadFromKaggle';

declare global {
  interface Window {
    episode?: any;
  }
}

let hasData = false;

export function HomePage(): JSX.Element {
  const episode = useStore(state => state.episode);
  const load = useStore(state => state.load);

  const navigate = useNavigate();

  const onWindowMessage = useCallback((event: MessageEvent<any>) => {
    if (hasData) {
      return;
    }

    if (event.data && event.data.environment) {
      hasData = true;

      try {
        load(event.data.environment);
        navigate('/leaderboard');
      } catch (err: any) {
        console.error(err);
        notifyError('Cannot load episode from Kaggle', err.message);
      }
    }
  }, []);

  useEffect(() => {
    window.addEventListener('message', onWindowMessage);
    return () => {
      window.removeEventListener('message', onWindowMessage);
    };
  });

  if (episode === null && window.episode !== undefined) {
    try {
      load(window.episode);
      return <Navigate to="/visualizer" />;
    } catch (err: any) {
      console.error(err);
      notifyError('Cannot load episode', err.message);
    }
  }

  return (
    <Container>
      <Stack mb="md">
        <HomeCard title="Welcome!">
          {/* prettier-ignore */}
          <Text>
            Lux Eye S2 is a visualizer for <a href={`https://www.kaggle.com/competitions/lux-ai-season-2`} target="_blank" rel="noreferrer">Lux AI Season 2</a> episodes.
            Load an episode below to get started.
          </Text>
        </HomeCard>

        <LoadFromFile />
        <LoadFromKaggle />
        <LoadFromElsewhere />
      </Stack>
    </Container>
  );
}
