import { useEffect, useState } from "react";
import type { ChangeEvent } from "react";
import type { AxiosError } from "axios";
import { api } from "./services/api";

import {
  Activity,
  Database,
  Search,
  Upload,
  Play,
  RefreshCw,
  Server,
  CheckCircle2,
  AlertCircle,
  Layers3,
  Zap,
  Clock3,
  Menu,
} from "lucide-react";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

type Dataset = {
  dataset_id: string;
  filename: string;
  collection_name: string;
  document_count: number;
  dimension: number;
  indexing_time_ms: number;
};

type SearchResult = {
  id: number;
  text: string;
  score: number;
};

type Benchmark = {
  benchmark_id: string;
  engine: string;
  dataset_id: string;
  created_at: string;
  metrics: {
    recall_at_k: number;
    average_latency_ms: number;
    p95_latency_ms: number;
    qps: number;
    backend_process_ram_mb: number;
  };
};

const engines = [
  "Qdrant",
  "Milvus",
  "Weaviate",
  "Chroma",
  "FAISS",
  "pgvector",
  "Pinecone",
];

function getApiError(error: unknown, fallback: string): string {
  const axiosError = error as AxiosError<{ detail?: string }>;
  return axiosError.response?.data?.detail ?? fallback;
}

function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [selectedEngine, setSelectedEngine] = useState("qdrant");
  const [selectedDataset, setSelectedDataset] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState("Vérification...");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const selectedEngineLabel =
    engines.find((engine) => engine.toLowerCase() === selectedEngine) ||
    "Qdrant";

  async function loadData() {
    try {
      const [datasetResponse, benchmarkResponse, statusResponse] =
        await Promise.all([
          api.get("/api/datasets"),
          api.get("/api/benchmarks/results"),
          api.get(`/api/engines/${selectedEngine}/status`),
          api.get("/api/engines/qdrant/status"),
        ]);

      setDatasets(datasetResponse.data.datasets || []);
      setBenchmarks(benchmarkResponse.data.results || []);

      setStatus(statusResponse.data.connected ? "Connecté" : "Déconnecté");
    } catch {
      setStatus("API indisponible");
    }
  }

  useEffect(() => {
    void loadData();
  }, [selectedEngine]);

  async function uploadDataset() {
    if (!file) {
      setError("Sélectionne d'abord un fichier CSV, JSONL ou TXT.");
      return;
    }

    setLoading(true);
    setError("");
    setMessage("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await api.post("/api/datasets", formData);

      setMessage(
        `Dataset importé : ${response.data.dataset.document_count} documents.`,
      );

      setFile(null);

      await loadData();

      setSelectedDataset(response.data.dataset.dataset_id);
    } catch (err: unknown) {
      setError(getApiError(err, "Erreur pendant l'importation."));
    } finally {
      setLoading(false);
    }
  }

  async function searchDocuments() {
    if (!query.trim() || !selectedDataset) {
      setError("Saisis une question et sélectionne un dataset.");
      return;
    }

    setLoading(true);
    setError("");
    setMessage("");

    try {
      const response = await api.post("/api/search", {
        query,
        limit: 5,
        dataset_id: selectedDataset,
        engine: selectedEngine,
      });

      setSearchResults(response.data.results || []);

      setMessage(`${response.data.total} résultat(s) trouvé(s).`);
    } catch (err: unknown) {
      setError(getApiError(err, "Erreur de recherche."));
    } finally {
      setLoading(false);
    }
  }

  async function runBenchmark() {
    if (!selectedDataset) {
      setError("Sélectionne un dataset avant le benchmark.");
      return;
    }

    setLoading(true);
    setError("");
    setMessage("");

    try {
      const response = await api.post("/api/benchmarks/run", {
        dataset_id: selectedDataset,
        engine: selectedEngine,
        queries: [
          "Comment fonctionne une base de données vectorielle ?",
          "Comment améliorer la vitesse de recherche ?",
          "Comment réduire la mémoire utilisée par les vecteurs ?",
        ],
        top_k: 10,
        repetitions: 10,
      });

      setMessage(
        `Benchmark terminé : Recall@10 = ${response.data.metrics.recall_at_k}.`,
      );

      await loadData();
    } catch (err: unknown) {
      setError(getApiError(err, "Erreur pendant le benchmark."));
    } finally {
      setLoading(false);
    }
  }

  const latestBenchmark = benchmarks[0];

  const chartData = benchmarks
    .slice(0, 8)
    .reverse()
    .map((item, index) => ({
      name: `Test ${index + 1}`,
      latency: item.metrics.average_latency_ms,
    }));

  const totalDocuments = datasets.reduce(
    (sum, item) => sum + item.document_count,
    0,
  );

  const selectedDatasetName =
    datasets.find((item) => item.dataset_id === selectedDataset)?.filename ||
    "Aucun dataset";

  const navItems = [
    { href: "#dashboard", label: "Vue générale", icon: Activity },
    { href: "#datasets", label: "Datasets", icon: Database },
    { href: "#search", label: "Recherche", icon: Search },
    { href: "#benchmarks", label: "Benchmarks", icon: Play },
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      {/* SIDEBAR */}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col bg-slate-950 px-5 py-7 text-white transition-transform duration-300 lg:translate-x-0 ${
          mobileMenuOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <a href="#dashboard" className="mb-10 flex items-center gap-3 px-2">
          <div className="grid h-12 w-12 place-items-center rounded-2xl bg-indigo-500 shadow-lg">
            <Database size={25} />
          </div>

          <div>
            <h2 className="text-xl font-bold">VectorBench</h2>
            <span className="text-xs text-slate-400">Benchmark Platform</span>
          </div>
        </a>

        <p className="mb-3 px-3 text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
          Workspace
        </p>

        <nav className="grid gap-2">
          {navItems.map(({ href, label, icon: Icon }, index) => (
            <a
              key={href}
              href={href}
              onClick={() => setMobileMenuOpen(false)}
              className={`flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition ${
                index === 0
                  ? "bg-indigo-500 text-white"
                  : "text-slate-400 hover:bg-slate-800 hover:text-white"
              }`}
            >
              <Icon size={18} />
              {label}
            </a>
          ))}
        </nav>

        <div className="mt-auto rounded-2xl border border-slate-800 bg-slate-900 p-4">
          <div className="mb-4 flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300">
              Moteurs disponibles
            </span>
            <Layers3 size={16} />
          </div>

          <div className="space-y-3">
            {engines.map((engine) => (
              <div
                key={engine}
                className="flex items-center gap-2.5 text-xs text-slate-400"
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    engine === "Qdrant" ? "bg-emerald-400" : "bg-slate-600"
                  }`}
                />

                {engine}

                {engine === "Qdrant" && (
                  <span className="ml-auto text-emerald-300">Actif</span>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="mt-5 flex items-center gap-2 px-2 text-xs text-slate-500">
          <Zap size={14} />
          Vector database analytics
        </div>
      </aside>

      {/* MOBILE OVERLAY */}

      {mobileMenuOpen && (
        <button
          aria-label="Fermer le menu"
          onClick={() => setMobileMenuOpen(false)}
          className="fixed inset-0 z-30 bg-slate-950/50 lg:hidden"
        />
      )}

      {/* MAIN */}

      <main id="dashboard" className="min-h-screen lg:ml-72">
        {/* HEADER */}

        <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 px-5 py-4 backdrop-blur-xl sm:px-8 lg:px-10">
          <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setMobileMenuOpen(true)}
                className="rounded-xl border border-slate-200 p-2 text-slate-600 lg:hidden"
                aria-label="Ouvrir le menu"
              >
                <Menu size={20} />
              </button>

              <div>
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-indigo-500">
                  Plateforme d'analyse
                </p>

                <h1 className="mt-1 text-xl font-bold text-slate-900 sm:text-2xl">
                  Tableau de bord
                </h1>

                <p className="mt-1 hidden text-sm text-slate-500 sm:block">
                  Gère tes datasets et analyse les performances vectorielles.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-3 py-2 shadow-sm">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  status === "Connecté" ? "bg-emerald-500" : "bg-rose-500"
                }`}
              />

              <span className="hidden text-xs font-semibold text-slate-600 sm:inline">
                {selectedEngineLabel} : {status}
              </span>

              <button
                onClick={() => void loadData()}
                title="Actualiser"
                className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 hover:text-indigo-600"
              >
                <RefreshCw size={17} />
              </button>
            </div>
          </div>
        </header>

        <div className="mx-auto max-w-[1600px] px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
          {/* NOTIFICATIONS */}

          {(message || error) && (
            <div
              className={`mb-6 flex items-start gap-3 rounded-xl border px-4 py-3 text-sm ${
                error
                  ? "border-rose-200 bg-rose-50 text-rose-700"
                  : "border-emerald-200 bg-emerald-50 text-emerald-700"
              }`}
            >
              {error ? <AlertCircle size={18} /> : <CheckCircle2 size={18} />}

              <span>{error || message}</span>
            </div>
          )}

          {/* STATISTICS */}

          <section className="mb-7 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {[
              {
                label: "Datasets importés",
                value: datasets.length,
                icon: Database,
                tone: "bg-indigo-50 text-indigo-600",
              },
              {
                label: "Documents stockés",
                value: totalDocuments.toLocaleString("fr-FR"),
                icon: Server,
                tone: "bg-sky-50 text-sky-600",
              },
              {
                label: "Benchmarks réalisés",
                value: benchmarks.length,
                icon: Activity,
                tone: "bg-violet-50 text-violet-600",
              },
              {
                label: "Dernière latence moyenne",
                value: latestBenchmark
                  ? `${latestBenchmark.metrics.average_latency_ms} ms`
                  : "—",
                icon: Clock3,
                tone: "bg-amber-50 text-amber-600",
              },
            ].map(({ label, value, icon: Icon, tone }) => (
              <div
                key={label}
                className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-slate-500">
                    {label}
                  </span>

                  <span
                    className={`grid h-10 w-10 place-items-center rounded-xl ${tone}`}
                  >
                    <Icon size={20} />
                  </span>
                </div>

                <strong className="mt-4 block text-2xl font-bold text-slate-900">
                  {value}
                </strong>
              </div>
            ))}
          </section>

          {/* DATASETS */}

          <section
            id="datasets"
            className="mb-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7"
          >
            <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-indigo-500">
                  Gestion des données
                </p>

                <h2 className="mt-1 text-lg font-bold text-slate-900">
                  Importer un dataset
                </h2>

                <div className="mt-5">
                  <label
                    htmlFor="engine-select"
                    className="mb-2 block text-sm font-semibold text-slate-700"
                  >
                    Sélection du moteur vectoriel
                  </label>

                  <select
                    id="engine-select"
                    value={selectedEngine}
                    onChange={(event) => {
                      setSelectedEngine(event.target.value);
                      setSearchResults([]);
                      setMessage("");
                      setError("");
                    }}
                    className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 outline-none transition focus:border-indigo-400 focus:ring-4 focus:ring-indigo-500/10"
                  >
                    <option value="qdrant">Qdrant</option>
                    <option value="milvus">Milvus</option>
                    <option value="weaviate">Weaviate</option>
                    <option value="chroma">Chroma</option>
                    <option value="faiss">FAISS</option>
                    <option value="pgvector">pgvector</option>
                    <option value="pinecone">Pinecone</option>
                  </select>

                  <p className="mt-2 text-xs text-slate-500">
                    Les recherches et les benchmarks seront exécutés avec ce
                    moteur.
                  </p>
                </div>

                <p className="mt-1 text-sm text-slate-500">
                  Formats acceptés : CSV, JSONL et TXT (20 Mo maximum).
                </p>
              </div>

              <span className="grid h-11 w-11 place-items-center rounded-xl bg-indigo-50 text-indigo-600">
                <Upload size={21} />
              </span>
            </div>

            <div className="flex flex-col gap-3 rounded-2xl border border-dashed border-indigo-200 bg-indigo-50/50 p-4 sm:flex-row sm:items-center sm:justify-between sm:p-5">
              <div className="min-w-0">
                <label
                  htmlFor="dataset-file"
                  className="mb-2 block text-xs font-semibold text-slate-700"
                >
                  Choisir un fichier
                </label>

                <input
                  id="dataset-file"
                  type="file"
                  accept=".csv,.jsonl,.txt"
                  onChange={(event: ChangeEvent<HTMLInputElement>) =>
                    setFile(event.target.files?.[0] || null)
                  }
                  className="block w-full text-xs text-slate-600 file:mr-3 file:rounded-lg file:border-0 file:bg-white file:px-3 file:py-2 file:text-xs file:font-semibold file:text-indigo-700"
                />

                {file && (
                  <p className="mt-2 truncate text-xs text-slate-500">
                    {file.name}
                  </p>
                )}
              </div>

              <button
                onClick={() => void uploadDataset()}
                disabled={loading || !file}
                className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-indigo-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Upload size={17} />
                {loading ? "Traitement..." : "Importer le fichier"}
              </button>
            </div>

            <div className="mt-8 flex items-center justify-between gap-3">
              <h3 className="font-semibold text-slate-900">
                Datasets disponibles
              </h3>

              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
                {datasets.length} dataset(s)
              </span>
            </div>

            <div className="mt-4 overflow-x-auto rounded-xl border border-slate-200">
              <table className="w-full min-w-[650px] text-left text-sm">
                <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="px-4 py-3 font-semibold">Fichier</th>
                    <th className="px-4 py-3 font-semibold">Documents</th>
                    <th className="px-4 py-3 font-semibold">Dimension</th>
                    <th className="px-4 py-3 font-semibold">Indexation</th>
                    <th className="px-4 py-3 font-semibold">Utiliser</th>
                  </tr>
                </thead>

                <tbody className="divide-y divide-slate-100">
                  {datasets.map((dataset) => (
                    <tr
                      key={dataset.dataset_id}
                      className="transition hover:bg-slate-50"
                    >
                      <td className="px-4 py-4 font-medium text-slate-800">
                        {dataset.filename}
                      </td>

                      <td className="px-4 py-4 text-slate-600">
                        {dataset.document_count}
                      </td>

                      <td className="px-4 py-4 text-slate-600">
                        {dataset.dimension}
                      </td>

                      <td className="px-4 py-4 text-slate-600">
                        {dataset.indexing_time_ms} ms
                      </td>

                      <td className="px-4 py-4">
                        <button
                          onClick={() => setSelectedDataset(dataset.dataset_id)}
                          className={`rounded-lg px-3 py-2 text-xs font-semibold transition ${
                            selectedDataset === dataset.dataset_id
                              ? "bg-emerald-50 text-emerald-700"
                              : "bg-indigo-50 text-indigo-700 hover:bg-indigo-100"
                          }`}
                        >
                          {selectedDataset === dataset.dataset_id
                            ? "✓ Sélectionné"
                            : "Sélectionner"}
                        </button>
                      </td>
                    </tr>
                  ))}

                  {datasets.length === 0 && (
                    <tr>
                      <td
                        colSpan={5}
                        className="px-4 py-10 text-center text-sm text-slate-400"
                      >
                        Aucun dataset importé.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          {/* SEARCH AND BENCHMARK */}

          <section className="mb-6 grid gap-6 xl:grid-cols-2">
            {/* SEARCH */}

            <div
              id="search"
              className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7"
            >
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-indigo-500">
                    Recherche sémantique
                  </p>

                  <h2 className="mt-1 text-lg font-bold text-slate-900">
                    Tester une recherche
                  </h2>
                </div>

                <span className="grid h-11 w-11 place-items-center rounded-xl bg-indigo-50 text-indigo-600">
                  <Search size={21} />
                </span>
              </div>

              <label
                htmlFor="search-query"
                className="mb-2 block text-sm font-semibold text-slate-700"
              >
                Question à rechercher
              </label>

              <textarea
                id="search-query"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Exemple : comment améliorer la vitesse d'une recherche vectorielle ?"
                className="min-h-32 w-full resize-y rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700 outline-none transition placeholder:text-slate-400 focus:border-indigo-400 focus:bg-white focus:ring-4 focus:ring-indigo-500/10"
              />

              <button
                onClick={() => void searchDocuments()}
                disabled={loading}
                className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:opacity-50"
              >
                <Search size={17} />
                Rechercher
              </button>

              <div className="mt-5 space-y-3">
                {searchResults.map((result) => (
                  <article
                    key={result.id}
                    className="rounded-xl border border-slate-100 bg-slate-50 p-4"
                  >
                    <span className="text-xs font-bold text-indigo-600">
                      Score : {result.score.toFixed(4)}
                    </span>

                    <p className="mt-2 text-sm leading-6 text-slate-600">
                      {result.text}
                    </p>
                  </article>
                ))}

                {searchResults.length === 0 && (
                  <p className="rounded-xl bg-slate-50 px-4 py-5 text-center text-xs text-slate-400">
                    Les résultats de recherche apparaîtront ici.
                  </p>
                )}
              </div>
            </div>

            {/* BENCHMARK */}

            <div
              id="benchmarks"
              className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7"
            >
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-indigo-500">
                    Évaluation
                  </p>

                  <h2 className="mt-1 text-lg font-bold text-slate-900">
                    Lancer un benchmark
                  </h2>
                </div>

                <span className="grid h-11 w-11 place-items-center rounded-xl bg-violet-50 text-violet-600">
                  <Play size={21} />
                </span>
              </div>

              <p className="text-sm leading-6 text-slate-500">
                Le test utilise trois requêtes et dix répétitions par requête.
                Sélectionne d'abord un dataset.
              </p>

              <div className="mt-6 rounded-xl border border-slate-100 bg-slate-50 p-4">
                <span className="text-xs font-medium text-slate-500">
                  Dataset sélectionné
                </span>

                <strong className="mt-1 block truncate text-sm text-slate-800">
                  {selectedDatasetName}
                </strong>
              </div>

              <button
                onClick={() => void runBenchmark()}
                disabled={loading || !selectedDataset}
                className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Play size={17} />
                {loading ? "Exécution..." : "Lancer le benchmark"}
              </button>

              {latestBenchmark ? (
                <div className="mt-6 rounded-xl border border-emerald-100 bg-emerald-50/70 p-5">
                  <span className="text-xs font-semibold text-emerald-700">
                    Dernier Recall@10
                  </span>

                  <strong className="mt-1 block text-3xl font-bold text-emerald-800">
                    {(latestBenchmark.metrics.recall_at_k * 100).toFixed(1)} %
                  </strong>

                  <div className="mt-4 grid grid-cols-2 gap-3 text-xs text-emerald-800">
                    <span>
                      Latence :{" "}
                      <b>{latestBenchmark.metrics.average_latency_ms} ms</b>
                    </span>

                    <span>
                      QPS : <b>{latestBenchmark.metrics.qps}</b>
                    </span>
                  </div>
                </div>
              ) : (
                <p className="mt-6 rounded-xl bg-slate-50 p-5 text-center text-xs text-slate-400">
                  Le résultat du dernier benchmark apparaîtra ici.
                </p>
              )}
            </div>
          </section>

          {/* CHART */}

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
            <div className="mb-6">
              <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-indigo-500">
                Historique
              </p>

              <h2 className="mt-1 text-lg font-bold text-slate-900">
                Évolution des latences
              </h2>

              <p className="mt-1 text-sm text-slate-500">
                Comparaison des derniers benchmarks enregistrés.
              </p>
            </div>

            <div className="h-[280px] w-full">
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={chartData}
                    margin={{
                      top: 10,
                      right: 10,
                      left: 0,
                      bottom: 0,
                    }}
                  >
                    <CartesianGrid
                      stroke="#e2e8f0"
                      strokeDasharray="4 4"
                      vertical={false}
                    />

                    <XAxis
                      dataKey="name"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#64748b", fontSize: 12 }}
                    />

                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#64748b", fontSize: 12 }}
                    />

                    <Tooltip
                      cursor={{ fill: "#eef2ff" }}
                      contentStyle={{
                        borderRadius: 12,
                        border: "1px solid #e2e8f0",
                      }}
                    />

                    <Bar
                      dataKey="latency"
                      name="Latence (ms)"
                      fill="#6366f1"
                      radius={[7, 7, 0, 0]}
                      maxBarSize={54}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="grid h-full place-items-center rounded-xl bg-slate-50 text-center text-sm text-slate-400">
                  Les graphiques apparaîtront après le premier benchmark.
                </div>
              )}
            </div>
          </section>

          {/* FOOTER */}

          <footer className="py-8 text-center text-xs text-slate-400">
            VectorBench · Projet de comparaison des bases de données
            vectorielles
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
