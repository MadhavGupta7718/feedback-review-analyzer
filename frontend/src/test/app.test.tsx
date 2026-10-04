import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { fixtures, mockFetch } from "./mockApi";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

const LEAK = {
  email: /[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}/,
  url: /https?:\/\/|\bwww\.[a-z0-9]/i,
  prefixedId: /\b(?:ORD|ACC|CUST)-?\d{4,}/i,
  phoneLike: /(?<![\d-])\+?\d{3}[ .-]?\d{3}[ .-]?\d{4}(?![\d-])/,
};

function expectNoPii(text: string) {
  for (const [name, re] of Object.entries(LEAK)) {
    expect(re.test(text), `${name} pattern found in rendered page`).toBe(false);
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Executive Overview", () => {
  it("shows headline numbers and emerging complaints from the API", async () => {
    mockFetch();
    renderAt("/");
    expect(await screen.findByRole("heading", { name: "Executive Overview" })).toBeInTheDocument();
    const ov = fixtures.metrics.overview;
    expect(screen.getByText(ov.total_reviews.toLocaleString("en-US"))).toBeInTheDocument();
    expect(screen.getByText(`${ov.sentiment_pct.negative.toFixed(1)}%`)).toBeInTheDocument();
    for (const e of ov.emerging) expect(screen.getAllByText(e.name).length).toBeGreaterThan(0);
  });

  it("shows an error state with retry when the API is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    renderAt("/");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Cannot reach the analytics API");
    expect(within(alert).getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("shows the backend's generic message on a 503", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ error: "service_unavailable", detail: "Analytics data is not available." }), { status: 503 })),
    );
    renderAt("/");
    expect(await screen.findByRole("alert")).toHaveTextContent("Analytics data is not available.");
  });
});

describe("Complaint Radar", () => {
  it("VIEW WHY opens the calculation, reasons and evidence for an issue", async () => {
    mockFetch();
    renderAt("/radar");
    const first = fixtures.issues.issues[0];
    const row = (await screen.findAllByText(first.name))[0].closest("tr")!;
    await userEvent.click(within(row).getByRole("button", { name: "View why" }));
    const panel = await screen.findByTestId("why-panel");
    const detail = fixtures.issue;
    expect(within(panel).getByText(detail.calculation.growth)).toBeInTheDocument();
    expect(within(panel).getByText(detail.calculation.priority)).toBeInTheDocument();
    for (const r of detail.reasons) expect(within(panel).getByText(r)).toBeInTheDocument();
    const radarEvidence = fixtures.evidence.evidence.filter((e) => e.evidence_kind === "radar_evidence");
    await waitFor(() => expect(within(panel).getAllByTestId("review-card")).toHaveLength(radarEvidence.length));
  });

  it("every evidence review shown belongs to the issue's theme", async () => {
    mockFetch();
    renderAt(`/radar?issue=${fixtures.issue.theme_id}`);
    const panel = await screen.findByTestId("why-panel");
    await waitFor(() => expect(within(panel).getAllByTestId("review-card").length).toBeGreaterThan(0));
    for (const e of fixtures.evidence.evidence) expect(e.theme_id).toBe(fixtures.issue.theme_id);
  });
});

describe("Themes", () => {
  it("lists themes and opens a theme with representative reviews", async () => {
    mockFetch();
    renderAt("/themes");
    const t = fixtures.themes.themes.find((x) => x.theme_id === fixtures.theme.theme_id)!;
    await userEvent.click((await screen.findAllByText(t.name))[0]);
    await waitFor(() => expect(screen.getAllByTestId("review-card")).toHaveLength(fixtures.theme.representative_reviews.length));
  });

  it("filters themes by search text", async () => {
    mockFetch();
    renderAt("/themes");
    await screen.findByRole("heading", { name: "Themes" });
    await userEvent.type(screen.getByLabelText("Search themes"), "zzzz-no-match");
    expect(screen.getByText("No theme matches.")).toBeInTheDocument();
  });
});

describe("Evidence", () => {
  it("sends the search and filters to /reviews", async () => {
    const fetchFn = mockFetch();
    renderAt("/evidence");
    await screen.findByText(/matching reviews/);
    await userEvent.type(screen.getByLabelText("Search review text"), "battery");
    await userEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(fetchFn.mock.calls.some(([u]) => String(u).includes("/reviews?") && String(u).includes("q=battery"))).toBe(true));
  });

  it("opens a review drawer with probabilities", async () => {
    mockFetch();
    renderAt("/evidence");
    const id = fixtures.reviews.reviews[0].review_id;
    await userEvent.click(await screen.findByRole("button", { name: `Open review ${id}` }));
    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText(/Probabilities/)).toBeInTheDocument();
  });
});

describe("Model Validation", () => {
  it("shows global Amazon TEST accuracy and macro recall", async () => {
    mockFetch();
    renderAt("/sentiment");
    const acc = `${((fixtures.sentiment.metrics.headline_accuracy ?? 0) * 100).toFixed(1)}%`;
    expect((await screen.findAllByText(acc)).length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: "Model Validation" })).toBeInTheDocument();
    expect(screen.getAllByText(/held-out TEST split/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Cons_rating/i).length).toBeGreaterThan(0);
  });
});

describe("Upload & batches", () => {
  it("lists uploaded batches", async () => {
    mockFetch();
    renderAt("/batches");
    expect(await screen.findByRole("heading", { name: "Upload & batches" })).toBeInTheDocument();
    expect(screen.getByText("sample_reviews.csv")).toBeInTheDocument();
  });
});

describe("Data Health", () => {
  it("shows ingestion counts, traceability status and drift", async () => {
    mockFetch();
    renderAt("/health");
    expect(await screen.findByText(fixtures.dataHealth.input_rows.toLocaleString("en-US"))).toBeInTheDocument();
    expect(screen.getByText(fixtures.dataHealth.traceability!.status)).toBeInTheDocument();
    expect(await screen.findByText("Theme distribution")).toBeInTheDocument();
  });
});

describe("PII", () => {
  it.each(["/", "/themes?theme=" + fixtures.theme.theme_id, "/radar?issue=" + fixtures.issue.theme_id, "/evidence", "/health"])(
    "rendered page %s contains no PII patterns",
    async (path) => {
      mockFetch();
      renderAt(path);
      await waitFor(() => expect(screen.queryAllByRole("status")).toHaveLength(0));
      expectNoPii(document.body.textContent ?? "");
    },
  );
});
