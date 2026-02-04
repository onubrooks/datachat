export type WaitingUxMode = "basic" | "animated" | "progress";

const WAITING_UX_KEY = "datachat.waitingUxMode";

export const getWaitingUxMode = (): WaitingUxMode => {
  if (typeof window === "undefined") {
    return "animated";
  }
  const value = window.localStorage.getItem(WAITING_UX_KEY);
  if (value === "basic" || value === "progress" || value === "animated") {
    return value;
  }
  return "animated";
};

export const setWaitingUxMode = (mode: WaitingUxMode) => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(WAITING_UX_KEY, mode);
};
