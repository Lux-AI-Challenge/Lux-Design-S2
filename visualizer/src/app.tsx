import { useStore, useStoreKeys } from '@/store'
import { Landing } from './pages/Landing'
import { ReplayViewer } from './pages/ReplayViewer'
import './app.css'
import { useEffect, useState } from 'react'

export function App() {
  const replay = useStore((state) => state.replay)
  const { loadReplay } = useStoreKeys("loadReplay");
  //kaggle replay loader
  useEffect(() => {
    //@ts-ignore
    if (window.kaggle) {
      // check if window.kaggle.environment is valid and usable
      if (
        //@ts-ignore
        window.kaggle.environment &&
        //@ts-ignore
        window.kaggle.environment.steps.length > 1
      ) {
        console.log("Embedded kaggle replay detected, parsing it");
        //@ts-ignore
        let replay = window.kaggle.environment;
        loadReplay({type: "object", data: replay});
        // setReplay(2);
      } else {
        console.log(
          "Kaggle detected, but no replay, listening for postMessage"
        );
        // add this listener only once
        window.addEventListener(
          "message",
          (event) => {
            // Ensure the environment names match before updating.
            console.log({event});
            try {
              if (event.data.environment.name == "lux_ai_2022") {
                // updateContext(event.data);
                let replay = event.data.environment;
                // console.log("post message:");
                // console.log(event.data);
                loadReplay({type: "object", data: replay});
              }
            } catch (err) {
              console.error("Could not parse game");
              console.error(err);
            }
          },
          false
        );
      }
    }
  }, []);
  console.log("Re-render app", replay)
  return replay === null
    ? <Landing />
    : <ReplayViewer />
}