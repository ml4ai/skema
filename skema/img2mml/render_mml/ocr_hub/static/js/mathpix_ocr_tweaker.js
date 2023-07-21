/**
 * Showing different parts of the page
 */
const BASE_IMG_URL =
  "https://raw.githubusercontent.com/ml4ai/equation-images/main/images/";

async function show_original_image() {
  const image_id = data.id;
  $("#original-image-from-paper").attr("src", `${BASE_IMG_URL}${image_id}.png`);
}

async function show_image_info() {
  $("#img-count").text(`${cur_data_index + 1}/${total_data_size}`); // Image number

  // ID and confidence score
  const image_id_section = `<p><b>Image ID:</b> ${data.id}</p>`;
  const mathpix_confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  const mathpix_confidence_section = `<b> Mathpix Confidence Score:</b> ${mathpix_confidence}%`;
  $("#image-info").html(`${image_id_section}${mathpix_confidence_section}`);
}

async function show_latex() {
  const latex_code = data.latex;
  $("#latex-code").text(latex_code); // code
  $("#latex-render").text(`$$${data.latex}$$`); // render
}

async function show_mathml() {
  const mathml_code = data.mathml;
  $("#mathml-code").text(mathml_code); // code
  $("#mathml-render").html(mathml_code);
}

async function refresh_view() {
  $("#edit-latex-form")[0].reset();
  $("#edit-mathml-form")[0].reset();
  show_latex();
  show_mathml();
  show_image_info();
  show_original_image();
  update_btn_visibility();
  MathJax.typeset();
}

/**
 * Button Logic for changing the image and data currently being shown
 */
async function update_btn_visibility() {
  $("#prev-btn").addClass("btn-secondary");
  $("#next-btn").addClass("btn-secondary");

  if (cur_data_index > 0) {
    $("#prev-btn").addClass("btn-primary");
    $("#prev-btn").removeClass("btn-secondary");
  }
  if (cur_data_index < total_data_size - 1) {
    $("#next-btn").addClass("btn-primary");
    $("#next-btn").removeClass("btn-secondary");
  }
}

$("#prev-btn").click(async () => {
  if (cur_data_index > 0) {
    await fetch_data(cur_data_index - 1);
    refresh_view();
  }
});

$("#next-btn").click(async () => {
  if (cur_data_index < total_data_size - 1) {
    await fetch_data(cur_data_index + 1);
    refresh_view();
  }
});

/**
 * Preview changes being made to the code
 */
$("#latex-code").on("keyup", () => {
  const wip_latex_code = $("#latex-code").val();
  $("#latex-render").text(`$$${wip_latex_code}$$`);
  MathJax.typeset();
});

$("#mathml-code").on("keyup", () => {
  const wip_mathml_code = $("#mathml-code").val();
  $("#mathml-render").html(wip_mathml_code);
  MathJax.typeset();
});

/**
 * Save changes permanently
 */
$("#edit-latex-form").submit(async (e) => {
  e.preventDefault();
  if (confirm("Permanently save changes to LaTeX code?")) {
    data.latex = $("#latex-code").val();
    await post_data(cur_data_index, data);
    $("#edit-latex-form")[0].reset();
    refresh_view();
  }
});

$("#edit-mathml-form").submit(async (e) => {
  e.preventDefault();
  if (confirm("Permanently save changes to MathML code?")) {
    data.mathml = $("#mathml-code").val();
    await post_data(cur_data_index, data);
    $("#edit-mathml-form")[0].reset();
    refresh_view();
  }
});

/**
 * Handle Image Search
 */
$("#image-search").submit(async (e) => {
  e.preventDefault();
  let searched_id = $("#image-id").val();
  const searched_no = $("#image-no").val();

  if (searched_id !== "") {
    searched_id = searched_id.trim();
    // Search by ID if it was input
    await fetch_data(-1, searched_id);
    refresh_view();
    $("#image-search")[0].reset();
    return;
  }
  // Search by number if it was input
  if (searched_no > 0 && searched_no <= total_data_size) {
    await fetch_data(searched_no - 1);
    refresh_view();
    $("#image-search")[0].reset();
    return;
  }

  $("#image-search")[0].reset();
  alert("Invalid Image ID or Number Entered");
});

/**
 * Handle updating data between client and server
 */
async function fetch_data(index = 0, id = null) {
  let response = null;
  if (id) {
    response = await fetch(`/mathpix_annotation_data?id=${id}`);
  } else {
    response = await fetch(`/mathpix_annotation_data?index=${index}`);
  }

  response_data = await response.json();
  data = response_data.data[0];
  total_data_size = response_data.totalRecords;
  cur_data_index = response_data.index;
}

async function post_data(index = cur_data_index, updatedData = data) {
  const postData = { index, data: updatedData };
  await fetch("/mathpix_annotation_data/", {
    method: "POST",
    body: JSON.stringify(postData),
  });
}

/**
 * Initial View
 */
var cur_data_index = null;
var total_data_size = null;
var data = null;

$(document).ready(async () => {
  await fetch_data();
  refresh_view();
});
