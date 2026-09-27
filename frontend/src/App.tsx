import { lazy } from "react";
import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";

const Overview = lazy(() => import("./pages/Overview").then((m) => ({ default: m.Overview })));
const Themes = lazy(() => import("./pages/Themes").then((m) => ({ default: m.Themes })));
const Radar = lazy(() => import("./pages/Radar").then((m) => ({ default: m.Radar })));
const Evidence = lazy(() => import("./pages/Evidence").then((m) => ({ default: m.Evidence })));
const SentimentValidation = lazy(() => import("./pages/SentimentValidation").then((m) => ({ default: m.SentimentValidation })));
const DataHealth = lazy(() => import("./pages/DataHealth").then((m) => ({ default: m.DataHealth })));
const ProductBrief = lazy(() => import("./pages/ProductBrief").then((m) => ({ default: m.ProductBrief })));

export function App() {
  return (
    <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="themes" element={<Themes />} />
          <Route path="radar" element={<Radar />} />
          <Route path="evidence" element={<Evidence />} />
          <Route path="sentiment" element={<SentimentValidation />} />
          <Route path="health" element={<DataHealth />} />
          <Route path="brief" element={<ProductBrief />} />
          <Route path="*" element={<p className="page">Page not found.</p>} />
        </Route>
      </Routes>
  );
}
