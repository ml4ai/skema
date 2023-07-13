const BASE_IMG_URL =
  "https://raw.githubusercontent.com/imzoc/mathpix-annotation/master/mathml-images/images_filtered/";

async function init_table() {
  $("#model-results-table").DataTable({
    wordWrap: true,
    processing: true,
    serverSide: true,
    deferRender: true,
    scrollY: "78vh",
    scroller: true,
    ordering: false,
    pageLength: 10,
    info: true,
    stateSave: true,
    ajax: "/model_results",
    columns: [{ data: "id" }, { data: "id" }, { data: "mathml" }],
    columnDefs: [
      {
        targets: 0,
        width: "10%",
        searchable: true,
      },
      {
        targets: 1,
        searchable: false,
        width: "40%",
        render: function (data) {
          return `<img class="table-img" src="${BASE_IMG_URL}${data}.png"></img>`;
        },
      },
      {
        targets: 2,
        width: "50%",
        searchable: false,
      },
    ],
  });
}

$(document).ready(() => {
  init_table();
});
