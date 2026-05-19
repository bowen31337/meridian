import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.js";
import { MeridianApiProvider } from "./api/index.js";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
    },
  },
});

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const el = document.getElementById("root");
if (!el) throw new Error("Root element not found");

createRoot(el).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <MeridianApiProvider baseUrl={API_BASE_URL}>
        <App />
      </MeridianApiProvider>
    </QueryClientProvider>
  </StrictMode>,
);
