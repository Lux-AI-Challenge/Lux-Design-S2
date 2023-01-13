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
            Lux Eye 2022 is a visualizer for <a href={`https://www.kaggle.com/competitions/lux-ai-2022-beta`} target="_blank" rel="noreferrer">Lux AI 2022</a> episodes.
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
