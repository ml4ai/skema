// Assumes eqn_src has already been defined
// (loaded from latex_data_dev.js in render_equations.html)

// Version of table generation script that uses jquery to construct table rows

const format_xml = require("xml-formatter");

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

function build_row(id, source_img_path, latex_text, mml_text) {
  let row = $("<tr/>", { class: "datum", id: id });

  // Create a cell for each piece of data we want to show
  const id_text_cell = $("<td/>").append(`<p> ${id}</p>`);

  const source_img_cell = $("<td/>").append(
    `<img loading="lazy" src="${source_img_path}" width="auto" height="100%">`
  );

  const latex_text_cell = $("<td/>").append(`<pre>${latex_text}</pre>`);

  const latex_render_cell = $("<td/>").append(`<p>$$${latex_text}$$</p>`); // $$ allows mathjax to find and render it

  const mml_render_cell = $("<td/>", { class: "mml-render" }).append(mml_text);
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

  return row;
}

function add_table_rows(table) {
  for (element of eqn_src) {
    const id = element["id"];
    const source_img_path = `https://raw.githubusercontent.com/imzoc/mathpix-annotation/master/mathml-images/images_filtered/${id}.png`;
    const latex_text = element["latex"];
    // Adding <root> so that only one root tag exists later for XML formatting step
    const mml_text = `<root>${element["mathml"]}</root>`;

    row = build_row(id, source_img_path, latex_text, mml_text);
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
  // Ref: https://patdavid.net/2019/02/displaying-a-big-html-table/#:~:text=tl%3Bdr%20%2D%20To%20speed%20up,a%20table%20with%20~400%2C000%20cells.
  $("#table-loading").hide();
  table.show();
}

// triggers when HTML document is ready for processing
$(document).ready(function () {
  build_table(); // Build table into the HTML
});
