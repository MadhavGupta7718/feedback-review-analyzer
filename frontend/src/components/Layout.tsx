import { Suspense } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import { Loading } from "./ui";

const NAV = [
  { to: "/batches", label: "Upload & batches" },
  { to: "/sentiment", label: "Model Validation" },
  { to: "/", label: "Executive Overview", end: true },
  { to: "/themes", label: "Themes" },
  { to: "/radar", label: "Complaint Radar" },
  { to: "/evidence", label: "Evidence" },
  { to: "/health", label: "Data Health" },
];

export function Layout() {
  const health = useApi(() => api.health());
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            ◧
          </span>
          <div>
            <div className="brand-name">Review Analyzer</div>
            <div className="brand-sub">Upload batches · Amazon-trained sentiment</div>
          </div>
        </div>
        <nav>
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot" aria-live="polite">
          {health.loading ? (
            <span className="dot dot-grey" />
          ) : health.error || health.data?.status !== "ok" ? (
            <>
              <span className="dot dot-red" /> No active batch — upload a CSV
            </>
          ) : (
            <>
              <span className="dot dot-green" />
              {` API ok · ${(health.data.reviews ?? 0).toLocaleString("en-US")} reviews`}
              {health.data.dataset ? ` · ${health.data.dataset}` : ""}
            </>
          )}
        </div>
      </aside>
      <main className="main">
        <Suspense fallback={<Loading />}>
          <Outlet />
        </Suspense>
      </main>
    </div>
  );
}
