const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");

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
        ],
        devServer: {
            static: path.join(__dirname, "dist"),
            port: 8090,
            historyApiFallback: {
                rewrites: [{ from: /^\/stats/, to: "/stats/index.html" }],
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
