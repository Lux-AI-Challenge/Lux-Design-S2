import { Factory, Unit } from "../../types/replay/unit";
import s from "./unitslist.module.scss";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import useMediaQuery from '@mui/material/useMediaQuery';

import {
  Box,
  Divider,
  ListItemButton,
  Tab,
  Tabs,
  Typography,
} from "@mui/material";
import React, { useState } from "react";
import { Position } from "@/types/replay/position";
import { FrameStats, ReplayStats } from "@/types/replay";
import { Cargo } from "@/types/replay/cargo";
type UnitsListProps = {
  units: Record<string, Unit>;
  selectedUnits: Set<string>;
  factories: Record<string, Factory>;
  frameStats: FrameStats["player_0"];
  flex: boolean;
  agent: string;
};
function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    "aria-controls": `simple-tabpanel-${index}`,
  };
}
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 1.5 }}>
          <Typography>{children}</Typography>
        </Box>
      )}
    </div>
  );
}
function PosRep(pos: Position) {
  return (
    <>
      ({pos[0]}, {pos[1]})
    </>
  );
}
function CargoRep(cargo: Cargo) {
  return (
    <>
      <span>
        Ice: {cargo.ice}, Water: {cargo.water}
      </span>{" "}
      |{" "}
      <span>
        Ore: {cargo.ore}, Metal: {cargo.metal}
      </span>
    </>
  );
}
function None() {
  return (
    <><div className={s.noItem}>No units</div></>
  )
}
const listsx = { padding: "0rem" };
export const UnitsList = React.memo(
  ({ flex, frameStats, units, selectedUnits, factories, agent }: UnitsListProps) => {
    const [value, setValue] = useState(0);

    const handleChange = (event: React.SyntheticEvent, newValue: number) => {
      setValue(newValue);
    };
    const lights: Unit[] = [];
    const heavies: Unit[] = [];
    const factories_list: Factory[] = [];
    Object.entries(units).forEach(([unit_id, unit]) => {
      if (unit.unit_type == "LIGHT") {
        if (selectedUnits.has(unit.unit_id)) {
          lights.unshift(unit);
        } else {
          lights.push(unit);
        }
      } else {
        if (selectedUnits.has(unit.unit_id)) {
          heavies.unshift(unit);
        } else {
          heavies.push(unit);
        }
      }
    });
    Object.entries(factories).forEach(([unit_id, factory]) => {
      if (selectedUnits.has(factory.unit_id)) {
        factories_list.unshift(factory);
      } else {
        factories_list.push(factory);
      }
    });
    let color = "var(--p0_c)"
    if (agent == "player_1") {
      color = "var(--p1_c)"
    }
    const tabsx = { minHeight: "12px" };
    return (
      <>
        <div className={s.UnitsList} style={{width: flex ? "0" : "auto"}}>
          <Box
            className={s.box}
            sx={{ width: "100%", maxWidth: 360, bgcolor: "background.paper" }}
          >
            <Tabs
              value={value}
              onChange={handleChange}
              aria-label="unit tabs"
              TabIndicatorProps={{style:{background:color}}}
              sx={{
                m: 0,
                minHeight: "12px",
              }}
            >
              <Tab
                className={s.tabname}
                sx={tabsx}
                style={{color: color}}
                label={`Light (${lights.length})`}
                {...a11yProps(0)}
              />
              <Tab
                className={s.tabname}
                sx={tabsx}
                style={{color: color}}
                label={`Heavies (${heavies.length})`}
                {...a11yProps(1)}
              />
              <Tab
                className={s.tabname}
                sx={tabsx}
                style={{color: color}}
                label={`Factories (${factories_list.length})`}
                {...a11yProps(2)}
              />
            </Tabs>
            <TabPanel value={value} index={0}>
              <List sx={listsx} className={s.listWrapper}>
                {lights.map((unit) => {
                  const selected = selectedUnits.has(unit.unit_id);
                  const classname = selected
                    ? `${s.listItem} ${s.highlighted}`
                    : s.listItem;
                  return (
                    <>
                      {/* <ListItemButton> */}
                      <div className={classname}>
                        <div>
                          {unit.unit_id} {PosRep(unit.pos)}
                        </div>
                        <div className={s.attrs}>
                          Power: {unit.power} | {CargoRep(unit.cargo)}
                        </div>
                        <div></div>
                        <div></div>
                        <div></div>
                      </div>
                      {!selected && <Divider />}
                      {/* </ListItemButton> */}
                    </>
                  );
                })}
                {factories_list.length == 0 && None()}
              </List>
            </TabPanel>
            <TabPanel value={value} index={1}>
              <List sx={listsx} className={s.listWrapper}>
                {heavies.map((unit) => {
                  const selected = selectedUnits.has(unit.unit_id);
                  const classname = selected
                    ? `${s.listItem} ${s.highlighted}`
                    : s.listItem;
                  return (
                    <>
                      {/* <ListItemButton> */}
                      <div className={classname}>
                        <div>
                          {unit.unit_id} {PosRep(unit.pos)}
                        </div>
                        <div className={s.attrs}>
                          Power: {unit.power} | {CargoRep(unit.cargo)}
                        </div>
                        <div></div>
                        <div></div>
                        <div></div>
                      </div>
                      {!selected && <Divider />}
                      {/* </ListItemButton> */}
                    </>
                  );
                })}
                {factories_list.length == 0 && None()}
              </List>
            </TabPanel>
            <TabPanel value={value} index={2}>
              <List sx={listsx} className={s.listWrapper}>
                {factories_list.map((factory) => {
                  const selected = selectedUnits.has(factory.unit_id);
                  const classname = selected
                    ? `${s.listItem} ${s.highlighted}`
                    : s.listItem;
                  return (
                    <>
                      {/* <ListItemButton> */}
                      <div className={classname}>
                        <div>
                          {factory.unit_id} {PosRep(factory.pos)}
                        </div>
                        <div className={s.attrs}>
                          <div>
                            Power: {factory.power} | {CargoRep(factory.cargo)}
                          </div>
                          <div>
                            Lichen: {frameStats.factoryLichen[factory.unit_id]},
                            Connected Lichen Tiles:{" "}
                            {frameStats.factoryLichenTiles[factory.unit_id]}
                          </div>
                        </div>
                      </div>
                      {!selected && <Divider />}
                      {/* </ListItemButton> */}
                    </>
                  );
                })}
                {factories_list.length == 0 && None()}
              </List>
            </TabPanel>
          </Box>
        </div>
      </>
    );
  }
);
