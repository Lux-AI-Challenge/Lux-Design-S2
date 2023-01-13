import { parseLuxAI2022Episode } from './luxai2022';
import { Episode } from './model';

export function isKaggleEnvironmentsEpisode(data: any): boolean {
  return typeof data === 'object' && data.steps !== undefined;
}

export function parseKaggleEnvironmentsEpisode(data: any): Episode {
  const observations = [];
  const actions = [];

  let teamNames: [string, string] | undefined = undefined;
  if (typeof data.info === 'object' && data.info.TeamNames !== undefined) {
    teamNames = data.info.TeamNames;
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

  return parseLuxAI2022Episode({ observations, actions }, teamNames);
}
