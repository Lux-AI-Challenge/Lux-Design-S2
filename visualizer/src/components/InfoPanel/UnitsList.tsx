import { Factory, Unit } from "../../types/replay/unit";
import s from "./unitslist.module.scss";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import { Box, Divider, ListItemButton, Tab, Tabs, Typography } from "@mui/material";
import React, { useState } from "react";
import { Position } from "@/types/replay/position";
type UnitsListProps = {
  units: Record<string, Unit>;
  selectedUnit: string | null;
  factories: Record<string, Factory>;
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
        <Box sx={{ p: 3 }}>
          <Typography>{children}</Typography>
        </Box>
      )}
    </div>
  );
}
function PosRep(pos: Position) {
  return <>({pos[0]}, {pos[1]})</>
}

export const UnitsList = React.memo(({ units, selectedUnit, factories }: UnitsListProps) => {
  const [value, setValue] = useState(0);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };
  // <List>
  //         <ListItem disablePadding>
  //           <ListItemButton>
  //             <ListItemText primary="Trash" />
  //           </ListItemButton>
  //         </ListItem>
  //         <ListItem disablePadding>
  //           <ListItemButton component="a" href="#simple-list">
  //             <ListItemText primary="Spam" />
  //           </ListItemButton>
  //         </ListItem>
  //       </List>
  const lights: Unit[] = [];
  const heavies: Unit[] = [];
  const factories_list: Factory[] = [];
  Object.entries(units).forEach(([unit_id, unit]) => {
    if (unit.unit_type == "LIGHT") {
      lights.push(unit);
    } else{
      heavies.push(unit);
    }
  });
  Object.entries(factories).forEach(([unit_id, factory]) => {
    factories_list.push(factory);
  })
  const tabsx = {minHeight: "12px"}
  return (
    <>
      <div className={s.UnitsList}>
        <Box
          className={s.list}
          sx={{ width: "100%", maxWidth: 360, bgcolor: "background.paper" }}
        >
          <Tabs
            value={value}
            onChange={handleChange}
            aria-label="unit tabs"
            sx={{
              m:0,
              minHeight: "12px"
            }}
          >
            <Tab className={s.tabname} sx={tabsx} label={`Light (${lights.length})`} {...a11yProps(0)} />
            <Tab className={s.tabname} sx={tabsx} label={`Heavies (${heavies.length})`} {...a11yProps(1)} />
            <Tab className={s.tabname} sx={tabsx} label={`Factories (${factories_list.length})`} {...a11yProps(2)} />
          </Tabs>
          <TabPanel value={value} index={0}>
            <List>
              {lights
                .map((unit) => {
                  return (
                    <>
                      {/* <ListItemButton> */}
                      <div className={s.listItem}>
                        <div>{unit.unit_id} {PosRep(unit.pos)}</div>
                        <div className={s.power}>Power: {unit.power}</div>
                        <div></div>
                        <div></div>
                      </div>
                      <Divider/>
                      {/* </ListItemButton> */}
                    </>
                  );
                })}
            </List>
          </TabPanel>
          <TabPanel value={value} index={1}>
            <List>
              {heavies
                .map((unit) => {
                  return (
                    <>
                      {/* <ListItemButton> */}
                      <div className={s.listItem}>
                        <div>{unit.unit_id} {PosRep(unit.pos)}</div>
                        <div className={s.power}>{unit.power}</div>
                        <div>{unit.pos}</div>
                      </div>
                      <Divider/>
                      {/* </ListItemButton> */}
                    </>
                  );
                })}
            </List>
          </TabPanel>
          <TabPanel value={value} index={2}>
            <List>
              {factories_list
                .map((factory) => {
                  return (
                    <>
                      {/* <ListItemButton> */}
                      <div className={s.listItem}>
                        <div>{factory.unit_id} {PosRep(factory.pos)}</div>
                        <div className={s.power}>Power: {factory.power}</div>
                        {/* <div>{factory.pos}</div> */}
                      </div>
                      <Divider/>
                      {/* </ListItemButton> */}
                    </>
                  );
                })}
            </List>
          </TabPanel>
        </Box>
      </div>
    </>
  );
});