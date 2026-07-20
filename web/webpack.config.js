const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const HtmlPartialsPlugin = require("./webpack-partials");

// Multi-page app: the agent chat is the landing page (/), the stats dashboard is
// its own page (/stats/). Each page loads only its own entry bundle. The backend
// (FastAPI) serves the built pages from web/dist in production; in dev,
// webpack-dev-server serves them and proxies /api to the uvicorn backend.
module.exports = (env, argv) => {
    const isProd = argv.mode === "production";
    return {
        mode: isProd ? "production" : "development",
        entry: {
            agent: "./src/agent.ts",
            stats: "./src/stats.ts",
            settings: "./src/settings.ts",
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
            // Read-only agent settings/config at /settings/.
            new HtmlWebpackPlugin({
                template: "./src/settings.html",
                filename: "settings/index.html",
                chunks: ["settings"],
            }),
            // Inject the shared header/footer partials into both pages.
            new HtmlPartialsPlugin({ basePath: "/" }),
        ],
        devServer: {
            static: path.join(__dirname, "dist"),
            port: 8090,
            // Off by default `compress: true` runs the gzip middleware over the
            // whole pipeline, including proxied responses - and it BUFFERS
            // streaming bodies, so the /api/chat/stream SSE arrives in one lump
            // instead of token-by-token. Disable it so the dev server streams.
            compress: false,
            historyApiFallback: {
                rewrites: [
                    { from: /^\/stats/, to: "/stats/index.html" },
                    { from: /^\/settings/, to: "/settings/index.html" },
                ],
            },
            proxy: [
                {
                    context: ["/api"],
                    target:
                        process.env.SCUFRIS_API_URL || "http://localhost:8000",
                    changeOrigin: true,
                },
            ],
        },
        devtool: isProd ? false : "source-map",
    };
};
