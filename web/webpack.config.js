const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const HtmlPartialsPlugin = require("./webpack-partials");
const getPort = require("get-port");

// Load the repo-root .env (dev overrides like SCUFRIS_PORT) so `npm run serve`
// proxies /api to the SAME backend port the Python app reads from .env - one
// source of truth. webpack runs from web/, so reach up one dir. Node 20.12+/
// 21.7+ (the devshell nodejs); tolerate a missing file / older node.
try {
    process.loadEnvFile(path.resolve(__dirname, "..", ".env"));
} catch {
    // no .env or loadEnvFile unavailable - fall back to the defaults below
}

// Multi-page app: the agent chat is the landing page (/), the stats dashboard is
// its own page (/stats/). Each page loads only its own entry bundle. The backend
// (FastAPI) serves the built pages from web/dist in production; in dev,
// webpack-dev-server serves them and proxies /api to the uvicorn backend.
module.exports = async (env, argv) => {
    // Get a random port in 7XXX for the UI app; useful when having multiple things running in dev.
    try {
        process.env.SCUFRIS_UI_PORT = await getPort.default({
            port: getPort.portNumbers(7000, 7999),
        });
    } catch (e) {
        console.error("Failed to get a random port for SCUFRIS_UI_PORT:", e);
    }

    const SCUFRIS_UI_PORT = process.env.SCUFRIS_UI_PORT || 7000;
    const SCUFRIS_PORT = process.env.SCUFRIS_PORT || 7001;

    const isProd = argv.mode === "production";
    return {
        mode: isProd ? "production" : "development",
        entry: {
            agent: "./src/agent.ts",
            stats: "./src/stats.ts",
            settings: "./src/settings.ts",
            projects: "./src/projects.ts",
            agents: "./src/agents.ts",
            host: "./src/host.ts",
            "agent-detail": "./src/agent-detail.ts",
            "project-detail": "./src/project-detail.ts",
            login: "./src/login.ts",
        },
        output: {
            path: path.resolve(__dirname, "dist"),
            filename: "[name].js",
            publicPath: "/",
            clean: true,
        },
        resolve: {
            extensions: [".ts", ".js"],
            alias: {
                src: path.resolve(__dirname, "src"),
            },
        },
        module: {
            rules: [
                {
                    test: /\.ts$/,
                    use: "ts-loader",
                    exclude: /node_modules/,
                },
                {
                    test: /\.css$/i,
                    use: ["style-loader", "css-loader", "postcss-loader"],
                },
            ],
        },
        plugins: [
            // Landing page = agent chat, at /.
            new HtmlWebpackPlugin({
                template: "./src/index.html",
                filename: "index.html",
                chunks: ["agent"],
            }),
            // Stats dashboard at /stats/.
            new HtmlWebpackPlugin({
                template: "./src/stats.html",
                filename: "stats/index.html",
                chunks: ["stats"],
            }),
            // Host actions: the approval queue and the record, at /host/.
            new HtmlWebpackPlugin({
                template: "./src/host.html",
                filename: "host/index.html",
                chunks: ["host"],
            }),
            // Agent settings/operator console at /settings/.
            new HtmlWebpackPlugin({
                template: "./src/settings.html",
                filename: "settings/index.html",
                chunks: ["settings"],
            }),
            // Projects page at /projects/.
            new HtmlWebpackPlugin({
                template: "./src/projects.html",
                filename: "projects/index.html",
                chunks: ["projects"],
            }),
            // Agents dashboard at /agents/.
            new HtmlWebpackPlugin({
                template: "./src/agents.html",
                filename: "agents/index.html",
                chunks: ["agents"],
            }),
            // Per-agent detail SPA shell, served by the backend for /agents/<id>.
            new HtmlWebpackPlugin({
                template: "./src/agent-detail.html",
                filename: "agent-detail.html",
                chunks: ["agent-detail"],
            }),
            // Per-project detail SPA shell, served by the backend for /projects/<id>.
            new HtmlWebpackPlugin({
                template: "./src/project-detail.html",
                filename: "project-detail.html",
                chunks: ["project-detail"],
            }),
            // Sign-in page at /login/. Standalone: it is the ONE page reachable
            // without a session, so it must not depend on gated assets.
            new HtmlWebpackPlugin({
                template: "./src/login.html",
                filename: "login/index.html",
                chunks: ["login"],
            }),
            // Inject the shared header/footer partials into both pages.
            new HtmlPartialsPlugin({ basePath: "/" }),
        ],
        devServer: {
            static: path.join(__dirname, "dist"),
            port: SCUFRIS_UI_PORT,
            // Off by default `compress: true` runs the gzip middleware over the
            // whole pipeline, including proxied responses - and it BUFFERS
            // streaming bodies, so the /api/chat/stream SSE arrives in one lump
            // instead of token-by-token. Disable it so the dev server streams.
            compress: false,
            historyApiFallback: {
                rewrites: [
                    { from: /^\/login/, to: "/login/index.html" },
                    { from: /^\/stats/, to: "/stats/index.html" },
                    { from: /^\/host/, to: "/host/index.html" },
                    { from: /^\/settings/, to: "/settings/index.html" },
                    // A specific /projects/<id>[/...] goes to the detail shell;
                    // the bare /projects (list) falls through to its index below.
                    { from: /^\/projects\/.+/, to: "/project-detail.html" },
                    { from: /^\/projects/, to: "/projects/index.html" },
                    // A specific /agents/<id>[/...] goes to the detail shell;
                    // the bare /agents (list) falls through to its index below.
                    { from: /^\/agents\/.+/, to: "/agent-detail.html" },
                    { from: /^\/agents/, to: "/agents/index.html" },
                ],
            },
            proxy: [
                {
                    context: ["/api"],
                    target:
                        process.env.SCUFRIS_API_URL ||
                        `http://localhost:${SCUFRIS_PORT}`,
                    changeOrigin: true,
                },
            ],
        },
        devtool: isProd ? false : "source-map",
    };
};
