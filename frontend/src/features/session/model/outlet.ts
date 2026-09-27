import type { ChatSession } from "../../../api";

/** Context the user shell passes into the session page via the router outlet. */
export type AppOutlet = {
  sessions: ChatSession[];
  currentId: string | null;
  setCurrentId: (id: string | null) => void;
  refresh: (preferredId?: string) => Promise<string | null>;
  onNew: () => Promise<void>;
  collapsed: boolean;
  setCollapsed: (v: boolean) => void;
};
