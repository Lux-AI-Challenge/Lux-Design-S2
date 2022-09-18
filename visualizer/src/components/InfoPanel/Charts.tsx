import React, { useEffect, useState } from 'react';
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
  ReferenceLine,
} from 'recharts';
import { useStore, useStoreKeys } from '@/store';
const LineChartMemo = LineChart; //React.memo(LineChart);
const margin = { top: 15, right: 20, left: 10, bottom: 25 };
export const Charts = React.memo(() => {
  const replay = useStore((state) => state.replay)!; // game map should only get rendered when replay is non-null
  const { replayStats } = useStoreKeys(
    "replayStats"
  );
  const [data, setData]= useState<any>([]);
  useEffect(() => {
    // .slice(0, turn + 1)
    const d = replayStats.frameStats.map((v, i) => {
      return {name: `Turn ${i}`, p0light:v["player_0"].units.light, p1light:v["player_1"].units.light};
    });
    const labels = [];
    setData(d);
  }, []);
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
        <LineChartMemo
          // width={200}
          // onClick={() => {}}
          // height={150}
          data={data}
          margin={margin}
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
        </LineChartMemo>
      </ResponsiveContainer>
    </div>
    </>
  )
});