const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");

// Single-page dashboard. The backend (FastAPI) serves the built bundle from
// web/dist in production; in dev, webpack-dev-server serves it and proxies
// /api to the uvicorn backend so both hot-reload side by side.
module.exports = (env, argv) => {
    const isProd = argv.mode === "production";
    return {
        mode: isProd ? "production" : "development",
        entry: {
            main: "./src/main.ts",
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
            new HtmlWebpackPlugin({
                template: "./src/index.html",
                filename: "index.html",
            }),
        ],
        devServer: {
            static: path.join(__dirname, "dist"),
            port: 8090,
            historyApiFallback: true,
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
