import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  LineOptions,
} from 'chart.js';
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);
import s from "./charts.module.scss"
import {Line} from 'react-chartjs-2'
export const Charts = () => {
  // const labels = Utils.months({count: 7});
  const labels = [0,1,2,3,4,5,6,7,8,9,10];
  const defaultFontOptions = {
    color: 'white',
    font: {
      size: 14,
    }
  }
  const defaultOptions: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: 'white',
          font: {
            size: 14
          }
        }
      },
    },
    scales: {
      x: {
        title: {
          ...defaultFontOptions,
          display: true,
          text: 'Steps',
          font: {
            size: 14,
            weight: "600"
          }
        },
        grid: {
          color: 'rgba(255,255,255,0.2)',
          borderColor: 'rgba(255,255,255,0.5)',
        },
        ticks: {
          ...defaultFontOptions,
        }
      },
      y: {
        title: {
          ...defaultFontOptions,
          display: true,
          text: 'Count',
          font: {
            size: 14,
            weight: "600"
          }
        },
        grid: {
          color: 'rgba(255,255,255,0.2)',
          borderColor: 'rgba(255,255,255,0.5)',
        },
        beginAtZero: true,
        ticks: {
          ...defaultFontOptions,
        }
      }
    }
  };
  const data = {
    labels: labels,
    datasets: [{
      label: 'P0 Light Robots',
      data: [0, 1,10, 20],
      fill: false,
      borderColor: 'rgb(175, 192, 192)',
      tension: 0.1
    }, {
      label: 'P0 Heavy Robots',
      data: [65, 59, 80, 81, 56, 55, 40],
      fill: false,
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1
    },{
      label: 'P1 Light Robots',
      data: [1, 5, 21, 30, 55, 40],
      fill: false,
      borderColor: 'rgb(192, 75, 192)',
      tension: 0.1
    },{
      label: 'P1 Heavy Robots',
      data: [20, 30, 40],
      fill: false,
      borderColor: 'rgb(192, 175, 192)',
      tension: 0.1
    }]
  };
  return (
    <>
      <div>
        <h2>Robots</h2>
        <div className={s.chartWrapper}><Line options={{...defaultOptions}} data={data} /></div>
      </div>
    </>
  )
};
