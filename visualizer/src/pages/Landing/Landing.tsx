import { useRef, useCallback, useEffect } from "react";
import { useStoreKeys } from "@/store";

import uploadIcon from "@/assets/generic-icons/upload.svg";

import s from "./styles.module.scss";
import { Box, CircularProgress } from "@mui/material";

export function Landing() {
  const { progress, loadReplay } = useStoreKeys("progress", "loadReplay");

  const inputRef = useRef<HTMLInputElement>(null);

  const onButtonClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const handleUpload = useCallback(async () => {
    const file = inputRef.current?.files?.[0];
    if (!file) {
      return;
    }
    const name = file.name;
    const split = name.split(".");
    const extension = split.at(-1)!; // `String.split` always returns at least 1 length array
    if (extension === "json") {
      loadReplay({ type: "file", data: file });
    }
  }, []);

  //kaggle replay loader
  useEffect(() => {
    console.log("LOAD KAGGLE INFO?", {window}, window.kaggle)
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
                console.log("post message:");
                console.log(event.data);
                loadReplay({type: "object", data: replay});
                // const el = document.getElementsByTagName("html");
                // if (window.innerWidth * 0.65 <= 768) {
                //   el[0].style.fontSize = "6pt";
                // }
                // if (window.innerWidth * 0.65 <= 1280) {
                //   el[0].style.fontSize = "8pt";
                // }
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

  return (
    <div className={s.root}>
      {/* title */}
      <h1>Lux AI Season 2 Visualizer</h1>

      {/* upload replay button */}
      <input
        ref={inputRef}
        accept=".json, .luxr"
        type="file"
        onChange={handleUpload}
        style={{ display: "none" }}
      />
      <button onClick={onButtonClick} className={s.uploadButton}>
        <img src={uploadIcon} />
        upload replay
      </button>
      {progress !== null && (
        <span>
          <Box sx={{ display: "flex" }}>
            Loading...{" "}
            <CircularProgress sx={{ color: "white", ml: "0.5rem" }} size={24} />
          </Box>
        </span>
      )}
    </div>
  );
}
