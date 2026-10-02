/* Keep the current page when switching versions in the sidebar selector.
 *
 * sphinx_rtd_theme (>= 3.0) fills the <select> under the logo from the Read the Docs
 * Addons API ("readthedocs-addons-data-ready"), so the version list itself is native
 * and needs no maintenance here. The theme, however, links every option to the
 * *root* of the chosen version. Read the Docs' own flyout resolves the same page in
 * the other version instead (version root + the "readthedocs-resolver-filename"
 * meta tag that Read the Docs injects when serving the page), so this script applies
 * the same rule to the sidebar options. A page that does not exist in the chosen
 * version falls through to Read the Docs' 404 handling for that version.
 *
 * Outside Read the Docs (local builds) the event never fires and nothing happens.
 */
document.addEventListener("readthedocs-addons-data-ready", function (event) {
  const meta = document.querySelector('meta[name="readthedocs-resolver-filename"]');
  const filename = meta ? meta.getAttribute("content") : null;
  if (!filename) {
    return;
  }
  // The theme's own listener (registered later) builds the <select>; run after it.
  queueMicrotask(function () {
    const options = document.querySelectorAll(
      "div.switch-menus > div.version-switch select > option[data-url]",
    );
    const page = filename.replace(/\/index\.html$/, "/").replace(/^\//, "");
    options.forEach(function (option) {
      const root = option.dataset.url;
      if (root) {
        option.dataset.url = new URL(page, root.replace(/\/*$/, "/")).href;
      }
    });
  });
});
