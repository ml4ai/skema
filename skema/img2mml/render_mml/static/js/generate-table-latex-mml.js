// Assumes eqn_src has already been defined
// (loaded from latex_data_dev.js in render_equations.html)

// Version of table generation script that uses jquery to construct table rows

const format_xml = require("xml-formatter");

/**
 * Using the Googl Charts API to render LaTeX
 *
 * Ref:
 * https://stackoverflow.com/questions/33943355/js-script-to-convert-latex-formula-into-a-single-image
 * https://ardoris.wordpress.com/2010/06/27/converting-inline-latex-to-images-with-javascript-and-google-chart-api/
 */
function process_latex() {
  $("pre.latex").each(function (e) {
    var tex = $(this).text();
    var url =
      "http://chart.apis.google.com/chart?cht=tx&chl=" +
      encodeURIComponent(tex);
    var cls = $(this).attr("class");
    var img = '<img src="' + url + '" alt="' + tex + '" class="' + cls + '"/>';
    $(img).insertBefore($(this));
    $(this).hide();
  });
}

function add_table_headers(table) {
  const headers = [
    "ID",
    "Source Image",
    "LaTeX",
    "LaTeX Render",
    "MathML Render",
    "MathML",
  ];
  const header_row = $(
    `<tr>${headers.map((header) => {
      return `<th>${header}</th>`;
    })}</tr>`
  );
  table.append(header_row);
  return table;
}

function add_table_rows(table) {
  for (element of eqn_src) {
    let row = $("<tr/>", { class: "datum" });

    // Get info about the current entry
    const id = element["id"];
    const source_img_path = `https://raw.githubusercontent.com/imzoc/mathpix-annotation/master/mathml-images/images_filtered/${id}.${images_ext}`;
    const latex_text = element["latex"];
    // Adding a <root> tag around the mathml so that only one root exists later for
    // the XML formatting step instead of multiple <math> roots at the same level which does not work
    const mml_text = `<root>${element["mathml"]}</root>`;

    // Create a cell for each piece of data we want to show
    const id_text_cell = $("<td/>", {
      id: id,
      text: id,
    });

    const source_img_cell = $("<td/>").append(
      `<img src="${source_img_path}" width="auto" height="100%">`
    );

    const latex_text_cell = $("<td/>", {
      text: `${latex_text}`,
    });

    const latex_render_cell = $("<td/>").append(
      `<pre class="latex">${latex_text}</pre>`
    );

    const mml_render_cell = $("<td/>").append(mml_text);
    const mml_formatted_text = format_xml(mml_text, {
      indentation: "  ",
      collapseContent: true,
      lineSeparator: "\n",
    });
    const mml_text_cell = $("<td>").append(
      $("<div>", { class: "pre" }).append($("<pre>").text(mml_formatted_text))
    );

    // Add cells to the table
    row.append(id_text_cell);
    row.append(source_img_cell);
    row.append(latex_text_cell);
    row.append(latex_render_cell);
    row.append(mml_render_cell);
    row.append(mml_text_cell);
    table.append(row);
  }
  return table;
}

function build_table() {
  let table = $("#table");
  table.hide();
  table = add_table_headers(table);
  table = add_table_rows(table);

  // Show the table only after it has been built
  // This speeds up rendering by leaving it out of the render
  // tree until the last minute
  // Ref: https://patdavid.net/2019/02/displaying-a-big-html-table/#:~:text=tl%3Bdr%20%2D%20To%20speed%20up,a%20table%20with%20~400%2C000%20cells.
  $("#table-loading").hide();
  table.show();
}

// triggers when HTML document is ready for processing
$(document).ready(function () {
  build_table(); // Build table into the HTML
  process_latex(); // Render the Latex in the page (and the table) into an image
});
