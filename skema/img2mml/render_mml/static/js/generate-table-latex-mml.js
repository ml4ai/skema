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

function add_table_rows(table) {
  for ([i, element] of eqn_src.entries()) {
    let row = $("<tr/>", { class: "datum" });

    // Get info about the current entry
    const id = element["id"];
    const source_img = `${images_path}/source_imgs/${id}.${images_ext}`;
    const latex_text = element["latex"];
    const latex_render = `${images_path}/${i}.${images_ext}`;
    const mml_text = `${element["mathml"]}`;

    // Create a cell for each piece of data we want to show
    const id_text_cell = $("<td/>", {
      id: id,
      text: id,
    });

    const source_img_cell = $("<td/>").append(
      `<img src="${source_img}" width="auto" height="100%">`
    );

    const latex_text_cell = $("<td/>", {
      text: `${latex_text}`,
    });

    const latex_render_cell = $("<td/>").append(
      `<img src="${latex_render}" width="auto" height="100%">`
    );

    const mml_text_cell = $("<td/>", { id: `mml_img_${i}` }).html(mml_text);
    const mml_render = format_xml(mml_text, {
      indentation: "  ",
      collapseContent: true,
      lineSeparator: "\n",
    });
    const mml_render_cell = $("<td>", { id: `mml_src_${i}` }).append(
      $("<div>", { class: "pre" }).append($("<pre>").text(mml_render))
    );

    // Add cells to the table
    row.append(id_text_cell);
    row.append(source_img_cell);
    row.append(latex_text_cell);
    row.append(latex_render_cell);
    row.append(mml_text_cell);
    row.append(mml_render_cell);
    table.append(row);
  }
  return table;
}

function build_table() {
  let table = $("#table");
  table = add_table_headers(table);
  table = add_table_rows(table);
}

// triggers when HTML document is ready for processing
$(document).ready(function () {
  build_table();
});
