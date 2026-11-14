/** The control window's page: see App.tsx. */
import { render } from "solid-js/web";
import "../divora/tokens.css";
import "../divora/fonts.css";
import "../divora/components.css";
import "./control.css";
import { App } from "./App";

const root = document.getElementById("root");
if (root) render(() => <App />, root);
