import { Suspense } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import { Loading } from "./ui";

const NAV = [
  { to: "/", label: "Executive Overview", end: true },
  { to: "/themes", label: "Themes" },
  { to: "/radar", label: "Complaint Radar" },
  { to: "/evidence", label: "Evidence" },
  { to: "/sentiment", label: "Sentiment Validation" },
  { to: "/health", label: "Data Health" },
  { to: "/brief", label: "Product Brief" },
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
            <div className="brand-sub">10,000 reviews, no time to read them</div>
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
          ) : health.error ? (
            <>
              <span className="dot dot-red" /> API offline
            </>
          ) : (
            <>
              <span className="dot dot-green" /> API ok · {health.data?.reviews.toLocaleString("en-US")} reviews
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
