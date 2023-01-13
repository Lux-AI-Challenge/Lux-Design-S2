import { Paper } from '@mantine/core';
import { ApexOptions } from 'apexcharts';
import { useRef } from 'react';
import ReactApexChart from 'react-apexcharts';
import { Team } from '../../episode/model';
import { useStore } from '../../store';
import { getTeamColor } from '../../utils/colors';

export type ChartFunction = (team: Team) => number;

interface ChartProps {
  title: string;
  func: ChartFunction;
  step?: boolean;
}

export function Chart({ title, func, step }: ChartProps): JSX.Element {
  const chartRef = useRef<any>(null);

  const episode = useStore(state => state.episode)!;
  const steps = episode.steps.filter(step => step.step > 0);

  const exportFileName = title.replace(/\W/g, '_');

  const options: ApexOptions = {
    chart: {
      id: title,
      type: 'line',
      zoom: {
        autoScaleYaxis: true,
      },
      toolbar: {
        export: {
          csv: {
            filename: exportFileName,
          },
          svg: {
            filename: exportFileName,
          },
          png: {
            filename: exportFileName,
          },
        },
      },
      animations: {
        enabled: false,
      },
    },
    title: {
      text: title,
    },
    stroke: {
      width: 2,
      curve: step ? 'stepline' : 'smooth',
    },
    xaxis: {
      type: 'numeric',
    },
    yaxis: {
      labels: {
        formatter: value => value.toFixed(0),
      },
    },
    tooltip: {
      followCursor: true,
      x: {
        formatter: value => `Turn ${value}`,
      },
    },
  };

  const series: ApexAxisChartSeries = [];
  if (steps.length > 0) {
    for (let i = 0; i < steps[0].teams.length; i++) {
      series.push({
        name: steps[0].teams[i].name,
        data: steps.map(step => func(step.teams[i])),
        color: getTeamColor(i, 1.0),
      });
    }
  }

  return (
    <Paper shadow="xs" p="xs" withBorder={true}>
      <ReactApexChart ref={chartRef} options={options} series={series} height="300px" />
    </Paper>
  );
}
