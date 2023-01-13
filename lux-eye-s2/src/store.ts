import axios from 'axios';
import create from 'zustand';
import { isKaggleEnvironmentsEpisode, parseKaggleEnvironmentsEpisode } from './episode/kaggle-environments';
import { isLuxAI2022Episode, parseLuxAI2022Episode } from './episode/luxai2022';
import { Episode, Tile } from './episode/model';

export interface State {
  episode: Episode | null;
  rawEpisode: any | null;

  turn: number;
  speed: number;
  selectedTile: Tile | null;

  loading: boolean;
  progress: number;

  setTurn: (turn: number) => void;
  increaseTurn: () => boolean;
  setSpeed: (speed: number) => void;
  setSelectedTile: (selectedTile: Tile | null) => void;

  load: (data: any) => void;
  loadFromFile: (file: File) => Promise<void>;
  loadFromInput: (input: string, proxy: string) => Promise<void>;
}

export const useStore = create<State>((set, get) => ({
  episode: null,

  rawEpisode: null,

  turn: 1,
  speed: 1,
  selectedTile: null,

  loading: false,
  progress: 0,

  setTurn: turn => {
    if (get().turn !== turn) {
      set({ turn });
    }
  },

  increaseTurn: () => {
    if (get().turn === get().episode!.steps.length - 1) {
      return false;
    }

    set(state => ({ turn: state.turn + 1 }));
    return true;
  },

  setSpeed: speed => {
    if (get().speed !== speed) {
      set({ speed });
    }
  },

  setSelectedTile: selectedTile => {
    const current = get().selectedTile;
    if (
      (selectedTile === null && current !== null) ||
      (selectedTile !== null && (selectedTile.x !== current?.x || selectedTile.y !== current?.y))
    ) {
      set({ selectedTile });
    }
  },

  load: data => {
    const formatError =
      'Episode data has unsupported format, only JSON replays generated using the luxai2022 CLI or the kaggle-environments CLI are supported';

    if (typeof data !== 'object') {
      try {
        data = JSON.parse(data);
      } catch (err) {
        throw new Error(formatError);
      }
    }

    let episode: Episode | null = null;
    if (isLuxAI2022Episode(data)) {
      episode = parseLuxAI2022Episode(data);
    } else if (isKaggleEnvironmentsEpisode(data)) {
      episode = parseKaggleEnvironmentsEpisode(data);
    } else {
      throw new Error(formatError);
    }

    set({
      episode,
      rawEpisode: data,
      turn: 0,
      speed: 1,
    });
  },

  loadFromFile: file => {
    return new Promise((resolve, reject) => {
      set({ loading: true, progress: 0 });

      const reader = new FileReader();

      reader.addEventListener('load', () => {
        set({ loading: false });

        try {
          get().load(reader.result as string);
          resolve();
        } catch (err: any) {
          reject(err);
        }
      });

      reader.addEventListener('error', () => {
        reject(new Error('FileReader emitted an error event'));
      });

      reader.readAsText(file);
    });
  },

  loadFromInput: async (input, proxy) => {
    set({ loading: true, progress: 0 });

    const interestingPrefixes = [
      'https://www.kaggle.com/competitions/lux-ai-2022/leaderboard?dialog=episodes-episode-',
      'https://www.kaggle.com/competitions/lux-ai-2022-beta/leaderboard?dialog=episodes-episode-',
      'https://www.kaggle.com/competitions/lux-ai-2022/submissions?dialog=episodes-episode-',
      'https://www.kaggle.com/competitions/lux-ai-2022-beta/submissions?dialog=episodes-episode-',
    ];

    let url: string;
    if (/^\d+$/.test(input)) {
      url = `https://www.kaggleusercontent.com/episodes/${input}.json`;
    } else if (interestingPrefixes.some(prefix => input.startsWith(prefix))) {
      const id = input.split('-').pop();
      url = `https://www.kaggleusercontent.com/episodes/${id}.json`;
    } else {
      url = input;
    }

    let parsedURL: URL;
    try {
      parsedURL = new URL(url);
    } catch (err: any) {
      set({ loading: false });
      throw new Error('Invalid input');
    }

    if (parsedURL.hostname !== 'localhost' && proxy.trim().length > 0) {
      url = proxy + url;
    }

    try {
      const response = await axios.get(url, {
        onDownloadProgress: event => {
          if (event.loaded && event.total) {
            set({ progress: event.loaded / event.total });
          }
        },
      });

      set({ loading: false, progress: 0 });
      get().load(response.data);
    } catch (err: any) {
      set({ loading: false, progress: 0 });

      console.error(err);

      if (
        err.response &&
        typeof err.response.data === 'string' &&
        err.response.data.endsWith('was not whitelisted by the operator of this proxy.')
      ) {
        throw new Error('The current origin is not whitelisted by the operator of the specified CORS Anywhere proxy');
      }

      throw new Error(`${err.message}, see the browser console for more information`);
    }
  },
}));
