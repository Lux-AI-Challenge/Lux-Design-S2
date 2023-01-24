import { Paper, Title } from '@mantine/core';
import { ReactNode } from 'react';

interface HomeCardProps {
  title: string;
  children: ReactNode;
}

export function HomeCard({ title, children }: HomeCardProps): JSX.Element {
  return (
    <Paper shadow="xs" p="md" withBorder>
      <Title order={3} mb="xs">
        {title}
      </Title>

      {children}
    </Paper>
  );
}
