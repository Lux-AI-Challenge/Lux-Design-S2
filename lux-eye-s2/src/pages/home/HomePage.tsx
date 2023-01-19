import { Container, Stack, Text } from '@mantine/core';
import { HomeCard } from './HomeCard';
import { LoadFromElsewhere } from './LoadFromElsewhere';
import { LoadFromFile } from './LoadFromFile';
import { LoadFromKaggle } from './LoadFromKaggle';

export function HomePage(): JSX.Element {
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
