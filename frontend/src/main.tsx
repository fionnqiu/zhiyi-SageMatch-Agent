import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { ToastHost } from "./components/feedback/ToastHost";
import { applyTheme, readTheme } from "./lib/appearance/theme";
import "./index.css";

// CSS 加载前先落主题，避免浅色用户先闪一帧深色。
applyTheme(readTheme());

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
      <ToastHost />
    </BrowserRouter>
  </React.StrictMode>,
);
