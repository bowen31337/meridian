import { createBrowserRouter } from "react-router-dom";
import { AgentDetailPage } from "../pages/AgentDetailPage.js";
import { AgentsPage } from "../pages/AgentsPage.js";
import { ChannelsPage } from "../pages/ChannelsPage.js";
import { HomePage } from "../pages/HomePage.js";
import { MemoryStoresPage } from "../pages/MemoryStoresPage.js";
import { SessionDetailPage } from "../pages/SessionDetailPage.js";
import { SessionsPage } from "../pages/SessionsPage.js";
import { SettingsPage } from "../pages/SettingsPage.js";
import { SkillsPage } from "../pages/SkillsPage.js";
import { VaultsPage } from "../pages/VaultsPage.js";
import { ErrorPage } from "./ErrorPage.js";
import { NavigationTracer } from "./NavigationTracer.js";

export const router = createBrowserRouter([
  {
    element: <NavigationTracer />,
    errorElement: <ErrorPage />,
    children: [
      { path: "/", element: <HomePage /> },
      { path: "/sessions", element: <SessionsPage /> },
      { path: "/sessions/:id", element: <SessionDetailPage /> },
      { path: "/agents", element: <AgentsPage /> },
      { path: "/agents/:id", element: <AgentDetailPage /> },
      { path: "/skills", element: <SkillsPage /> },
      { path: "/channels", element: <ChannelsPage /> },
      { path: "/vaults", element: <VaultsPage /> },
      { path: "/memory_stores", element: <MemoryStoresPage /> },
      { path: "/settings", element: <SettingsPage /> },
    ],
  },
]);
