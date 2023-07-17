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
    columns: [
      { data: "id" },
      { data: "id" },
      { data: "mathml" },
      { data: "id" },
    ],
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
        render(data) {
          return `<img class="table-img" src="${BASE_IMG_URL}${data}.png"></img>`;
        },
      },
      {
        targets: 2,
        width: "45%",
        searchable: false,
      },
      {
        targets: 3,
        searchable: false,
        width: "5%",
        render(data) {
          return `
          <button
            type="button"
            class="btn btn-primary"
            data-bs-toggle="modal"
            data-bs-target="#edit-mathml-modal"
            onClick=populateEditModal('${data}')
          >
            Edit
          </button>`;
        },
      },
    ],
  });
}

$("#mathml-code").on("keyup", async () => {
  const wip_mathml_code = $("#mathml-code").val();
  $("#mathml-render").html(wip_mathml_code);
});

$("#save-mathml-code").on("click", async () => {
  if (confirm("Permanently save MathML changes?")) {
    const id = $("#edit-modal-image-id").val();
    const mathml_code = $("#mathml-code").val();
    post_data(id, mathml_code);
  }
});

async function fetch_data(id) {
  response = await fetch(
    `/model_results?draw=0&search[value]=${id}&search_exact=1`
  );
  const { data } = await response.json();
  if (data.length > 1) {
    throw ErrorEvent("Too many entries returned when only 1 was expected.");
  } else {
    return data[0];
  }
}

async function post_data(id, mathml) {
  await fetch("/model_results", {
    method: "POST",
    body: JSON.stringify({
      id,
      mathml,
    }),
  });
  refresh_table();
}

async function populateEditModal(id) {
  data = await fetch_data(id);
  const { mathml } = data;
  $("#edit-modal-image-id").val(id);
  $("#edit-modal-src-img").attr("src", `${BASE_IMG_URL}${id}.png`);
  $("#mathml-render").html(mathml);
  $("#mathml-code").val(mathml);
}

async function refresh_table() {
  $("#model-results-table").DataTable().ajax.reload();
}

$(document).ready(() => {
  init_table();
});
