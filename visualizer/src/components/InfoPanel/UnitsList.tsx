import { Factory, Unit } from "../../types/replay/unit";
import s from "./unitslist.module.scss";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import useMediaQuery from "@mui/material/useMediaQuery";
import InfoIcon from "@mui/icons-material/Info";
import {
  Box,
  Divider,
  IconButton,
  Tab,
  Tabs,
  Typography,
  Modal,
} from "@mui/material";
import React, { useState } from "react";
import { Position } from "@/types/replay/position";
import { FrameStats } from "@/types/replay";
import { Cargo } from "@/types/replay/cargo";
import { useStore, useStoreKeys } from "@/store";
type UnitsListProps = {
  units: Record<string, Unit>;
  selectedUnits: Set<string>;
  factories: Record<string, Factory>;
  frameStats: FrameStats["player_0"];
  flex: boolean;
  agent: string;
};
const style = {
  position: "absolute" as "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  width: 400,
  bgcolor: "background.paper",
  border: "2px solid #000",
  boxShadow: 24,
  color: "black",
  p: 4,
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
function DirectionRep(direction: number) {
  switch (direction) {
    case 0:
      return "center";
    case 1:
      return "up";
    case 2:
      return "right";
    case 3:
      return "down";
    case 4:
      return "left";
    default:
      return "unknown";
  }
}
function ResourceRep(resource: number) {
  switch (resource) {
    case 0:
      return "ice";
    case 1:
      return "ore";
    case 2:
      return "water";
    case 3:
      return "metal";
    case 4:
      return "power";
    default:
      return "unknown";
  }
}
function ActionRep(action: Array<number> | number) {
  if (typeof(action) === "number") {
    switch (action) {
      case 0: return <>Build Light Robot</>
      case 1: return <>Build Heavy Robot</>
      case 2: return <>Water</>
      default:
        return <>Unknown</>
    }
  }
  const tp = action[0];
  let actionType = "";
  let contents = "";
  switch (tp) {
    case 0:
      actionType = "Move";
      contents = DirectionRep(action[1]);
      break;
    case 1:
      actionType = "Transfer";
      contents = `${action[3]} units of ${ResourceRep(
        action[2]
      )} ${DirectionRep(action[1])}`;
      break;
    case 2:
      actionType = "Pickup";
      contents = `${action[3]} units of ${ResourceRep(action[2])}`;
      break;
    case 3:
      actionType = "Dig";
      break;
    case 4:
      actionType = "Self Destruct";
      break;
    case 5:
      actionType = "Recharge";
      contents = `until ${action[3]} power`;
      break;
    default:
      actionType = "Unknown";
  }
  if (action[4] == 1) {
    contents += " (repeat)";
  } else {
    contents += " (no repeat)";
  }
  return (
    <>
      {actionType} {contents}
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
    <>
      <div className={s.noItem}>No units</div>
    </>
  );
}
const listsx = { padding: "0rem" };
export const UnitsList = React.memo(
  ({
    flex,
    frameStats,
    units,
    selectedUnits,
    factories,
    agent,
  }: UnitsListProps) => {
    const replay = useStore((state) => state.replay)!;
    const { turn, speed, tileWidth } = useStoreKeys(
      "turn",
      "speed",
      "tileWidth"
    );
    const [value, setValue] = useState(0);
    const [unitInfoOpen, setUnitInfoOpen] = useState(false);
    const [viewedUnitInfo, setViewedUnitInfo] = useState<Unit | Factory>();
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
    let color = "var(--p0_c)";
    if (agent == "player_1") {
      color = "var(--p1_c)";
    }
    const tabsx = { minHeight: "12px" };
    const handleOpen = () => setUnitInfoOpen(true);
    const handleClose = () => setUnitInfoOpen(false);

    const infoButtonFactory = (unit_type: string, unit_id: string) => {
      return (
        <IconButton
          sx={{ p: 0.5, mt: -0.25 }}
          onClick={() => {
            if (unit_type == "robot") {
              setViewedUnitInfo(units[unit_id]);
            } else {
              setViewedUnitInfo(factories[unit_id]);
            }

            handleOpen();
          }}
        >
          <InfoIcon sx={{ width: "1.25rem", height: "1.25rem" }} />
        </IconButton>
      );
    };
    return (
      <>
        <div className={s.UnitsList} style={{ width: flex ? "0" : "auto" }}>
          <Modal
            open={unitInfoOpen}
            onClose={handleClose}
            aria-labelledby="modal-modal-title"
            aria-describedby="modal-modal-description"
          >
            <Box sx={style}>
              {viewedUnitInfo && (
                <>
                  <Typography id="unit-modal-title" variant="h6" component="h2">
                    Unit {viewedUnitInfo.unit_id} {PosRep(viewedUnitInfo.pos)}
                  </Typography>
                  <div className={s.modaldetails}>
                    <Typography id="unit-modal-description" sx={{ mt: 2 }}>
                      {viewedUnitInfo.action_queue &&
                      <>
                      <div>
                        <strong>Current Action Queue: </strong>
                      </div>
                      <div>
                        {
                          //@ts-ignore
                          viewedUnitInfo.action_queue.map(
                            (action: Array<number>, i: number) => {
                              return (
                                <div>
                                  {i}: {ActionRep(action)}
                                </div>
                              );
                            }
                          )
                        }
                      </div>
                      </>}
                      <div>
                        <strong>Actions executed in last 10 turns: </strong>
                      </div>
                      {replay.unitToActions[viewedUnitInfo.unit_id] &&
                        replay.unitToActions[viewedUnitInfo.unit_id]
                          .filter(
                            (data) => data.step > turn - 10 && data.step <= turn
                          )
                          .map((data) => {
                            let a = data.action;
                            if (typeof(data.action) !== "number") { // simple hack to check if viewedUnitInfo is a factory
                              a = data.action[0];
                            }
                            return (
                              <div>
                                {data.step - replay.meta.real_start}:{" "}
                                {ActionRep(a)}
                              </div>
                            );
                          })}
                    </Typography>
                  </div>
                </>
              )}
              {/* ActionRep */}
            </Box>
          </Modal>
          <Box
            className={s.box}
            sx={{ width: "100%", maxWidth: 360, bgcolor: "background.paper" }}
          >
            <Tabs
              value={value}
              onChange={handleChange}
              aria-label="unit tabs"
              TabIndicatorProps={{ style: { background: color } }}
              sx={{
                m: 0,
                minHeight: "12px",
              }}
            >
              <Tab
                className={s.tabname}
                sx={tabsx}
                style={{ color: color }}
                label={`Light (${lights.length})`}
                {...a11yProps(0)}
              />
              <Tab
                className={s.tabname}
                sx={tabsx}
                style={{ color: color }}
                label={`Heavies (${heavies.length})`}
                {...a11yProps(1)}
              />
              <Tab
                className={s.tabname}
                sx={tabsx}
                style={{ color: color }}
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
                          {unit.unit_id} {PosRep(unit.pos)}{" "}
                          {infoButtonFactory("robot", unit.unit_id)}
                          <IconButton>
                            <InfoIcon />
                          </IconButton>
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
                          {unit.unit_id} {PosRep(unit.pos)}{" "}
                          {infoButtonFactory("robot", unit.unit_id)}
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
                          {factory.unit_id} {PosRep(factory.pos)}{" "}
                          {infoButtonFactory("factory", factory.unit_id)}
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
