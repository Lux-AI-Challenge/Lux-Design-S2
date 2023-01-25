import { Stack, Title } from '@mantine/core';
import { IconCrown } from '@tabler/icons';
import { useStore } from '../../store';
import { getTeamColor } from '../../utils/colors';
import { getWinnerInfo } from './TeamCard';

interface TeamBannerProps {
  id: number;
  alignLeft: boolean;
}

export function TeamBanner({ id, alignLeft }: TeamBannerProps): JSX.Element {
  const episode = useStore(state => state.episode)!;
  const turn = useStore(state => state.turn);

  const step = episode.steps[turn];
  const team = step.teams[id];

  const winnerInfo = getWinnerInfo(episode, id);

  return (
    <Stack align={alignLeft ? 'flex-start' : 'flex-end'} spacing={0}>
      <Title order={4} style={{ color: getTeamColor(id, 1.0) }}>
        {winnerInfo[0] && <IconCrown style={{ verticalAlign: 'middle', marginRight: '2px' }} />}
        {team.name}
      </Title>
    </Stack>
  );
}
