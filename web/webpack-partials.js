const fs = require("fs");
const path = require("path");

// Injects shared header/footer partials into each generated page, replacing the
// <%= basePath %> placeholder so links resolve under the deploy base. A page
// opts in by including <div id="header"></div> / <div id="footer"></div>.
// Mirrors nova-protocol/web/webpack-partials.js (trimmed to header + footer).
class HtmlPartialsPlugin {
    constructor(options) {
        this.options = options || {};
    }

    apply(compiler) {
        compiler.hooks.compilation.tap("HtmlPartialsPlugin", (compilation) => {
            const HtmlWebpackPlugin = require("html-webpack-plugin");
            const hooks = HtmlWebpackPlugin.getHooks(compilation);

            hooks.beforeEmit.tapAsync(
                "HtmlPartialsPlugin",
                (data, callback) => {
                    const basePath =
                        data.plugin.options.basePath ||
                        this.options.basePath ||
                        "/";

                    const read = (name) => {
                        const p = path.join(__dirname, "src", name);
                        return fs.existsSync(p)
                            ? fs.readFileSync(p, "utf8")
                            : "";
                    };
                    const sub = (s) =>
                        s.replace(/<%=\s*basePath\s*%>/g, basePath);

                    data.html = data.html
                        .replace(
                            '<div id="header"></div>',
                            sub(read("_header.html")),
                        )
                        .replace(
                            '<div id="footer"></div>',
                            sub(read("_footer.html")),
                        );

                    callback(null, data);
                },
            );
        });
    }
}

module.exports = HtmlPartialsPlugin;
