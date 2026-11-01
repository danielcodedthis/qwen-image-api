// View routing through the URL hash: #create, #edit, #gallery, #gallery/<image id>.
// A refresh keeps the view, and the browser's Back button works.
import { emit } from "./events.js";

const VIEWS = ["create", "edit", "gallery"];

export function current() {
  const [view, ...rest] = location.hash.replace(/^#/, "").split("/");
  return {
    view: VIEWS.includes(view) ? view : "create",
    id: rest.length ? decodeURIComponent(rest.join("/")) : null,
  };
}

export function go(view, id = null) {
  const hash = "#" + view + (id ? "/" + encodeURIComponent(id) : "");
  if (location.hash === hash) emit("route", current());
  else location.hash = hash;
}

window.addEventListener("hashchange", () => emit("route", current()));
