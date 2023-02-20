import { parseLuxAIS2Episode } from './luxai-s2';
import { Episode, EpisodeMetadata } from './model';

export function isKaggleEnvironmentsEpisode(data: any): boolean {
  return typeof data === 'object' && data.steps !== undefined;
}

export function parseKaggleEnvironmentsEpisode(data: any): Episode {
  const observations = [];
  const actions = [];

  const extra: Partial<EpisodeMetadata> = {};
  if (typeof data.info === 'object' && data.info.TeamNames !== undefined) {
    extra.teamNames = data.info.TeamNames;
  }
  if (typeof data.configuration == 'object' && data.configuration.seed !== undefined) {
    extra.seed = data.configuration.seed;
  }

  for (const step of data.steps) {
    const obs = JSON.parse(step[0].observation.obs);

    observations.push(obs);
    actions.push({
      // eslint-disable-next-line @typescript-eslint/naming-convention
      player_0: step[0].action,
      // eslint-disable-next-line @typescript-eslint/naming-convention
      player_1: step[1].action,
    });
  }

  return parseLuxAIS2Episode({ observations, actions }, extra);
}
