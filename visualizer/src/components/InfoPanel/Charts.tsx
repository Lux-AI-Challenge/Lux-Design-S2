import React, { useEffect, useState } from 'react';
// import {
//   Chart as ChartJS,
//   CategoryScale,
//   LinearScale,
//   PointElement,
//   LineElement,
//   Title,
//   Tooltip,
//   Legend,
//   ChartOptions,
//   LineOptions,
// } from 'chart.js';
// ChartJS.register(
//   CategoryScale,
//   LinearScale,
//   PointElement,
//   LineElement,
//   Title,
//   Tooltip,
//   Legend
// );
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Label,
} from 'recharts';
import s from "./charts.module.scss"
import { useStore, useStoreKeys } from '@/store';
export const Charts = React.memo(() => {
  // const replay = useStore((state) => state.replay)!; // game map should only get rendered when replay is non-null
  const { turn, replayStats } = useStoreKeys(
    "turn",

    "replayStats"
  );
  const [labels, setLabels] = useState<Array<number>>([]);
  const [data, setData]= useState<any>([]);
  console.log("render")
  useEffect(() => {
    const d = replayStats.frameStats.slice(0, turn + 1).map((v, i) => {
      return {name: `Turn ${i}`, p0light:v["player_0"].units.light, p1light:v["player_1"].units.light};
    });
    const labels = [];
    for (let i = 0; i < data.length; i++) {
      labels.push(i);
    }
    setLabels(labels);
    setData(d);
  }, [turn]);
  const xlabel = "Turn"
  const ylabel = "Count";
  const TEAM_A_COLOR_STR = "#007D51";
  const TEAM_B_COLOR_STR = "#0082FB";
  return (
    <>
      {/* <div>
        <h2>Robots</h2>
        <div className={s.chartWrapper}><LineMemo options={{...defaultOptions}} data={data} /></div>
      </div> */}
      <div className="Graph">
      <ResponsiveContainer width={'100%'} height={240}>
        <LineChart
          // width={200}
          onClick={() => {}}
          // height={150}
          data={data}
          margin={{ top: 15, right: 20, left: 10, bottom: 25 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name">
            <Label
              value={xlabel}
              offset={-15}
              position="insideBottom"
              fill="#f9efe2"
            />
          </XAxis>
          <YAxis
            label={{
              value: ylabel,
              angle: -90,
              position: 'insideLeft',
              fill: '#f9efe2',
              color: 'f9efe2',
            }}
          ></YAxis>
          <Tooltip labelStyle={{ color: '#323D34' }} />
          {/* <Legend
            verticalAlign="top"
            height={36}
            formatter={renderColorfulLegendText}
          /> */}
          <Line
            type="monotone"
            dataKey="p0light"
            name="Player 0"
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
            stroke={TEAM_A_COLOR_STR}
          />
          <Line
            type="monotone"
            dataKey="p1light"
            name="Player 1"
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
            stroke={TEAM_B_COLOR_STR}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
    </>
  )
});